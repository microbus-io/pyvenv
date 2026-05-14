/*
Copyright (c) 2026 Microbus LLC and various contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package pyvenv runs a Python virtual environment as an in-process worker for a Go program.
// The Go side spawns one long-lived Python subprocess per Venv, talks to it over
// length-prefixed JSON frames on stdin/stdout, and exposes typed Call/Await into named Python
// functions.
//
// The module is framework-agnostic and has no third-party Go dependencies. The intended
// consumer is any Go program that wants to call into Python libraries (PyTorch, pandas,
// scikit-learn, etc.) without reimplementing subprocess management, framing, and result
// tracking.
//
// Call/Await are split so that a long-running Python computation can outlive the caller's context.
// Call enqueues a function invocation and returns a callID without waiting; the Python work runs
// in the worker until completion. The returned callID is the handle for Await(ctx, callID,
// &result), which blocks until the worker finishes or ctx expires. ctx expiry on Await does not
// cancel or consume the in-flight call — a subsequent Await(callID) by the same caller (or a
// retry of a workflow task that persisted the callID) re-enters the wait. A successful Await
// consumes the result; a subsequent Await on the same callID returns ErrUnknown.
//
// Typical lifecycle:
//
//  1. Embed Python sources at compile time:
//     //go:embed *.py
//     var pyFiles embed.FS
//
//  2. Build a Venv and Start it:
//     v, err := pyvenv.New(pyvenv.Config{
//     Sources:      ReadAll(pyFiles),
//     Requirements: []string{"pandas==2.2.0"},
//     MaxWorkers:   4,
//     })
//     err = v.Start(ctx)
//
//  3. Call into Python and Await the result:
//     callID, err := v.Call(ctx, "embed", map[string]any{"text": "hello"})
//     var out struct{ Vector []float64 }
//     err = v.Await(ctx, callID, &out)
//
//     Or use the synchronous convenience helper:
//     err = v.CallAndAwait(ctx, "embed", map[string]any{"text": "hello"}, &out)
//
//  4. Tear down:
//     v.Close(ctx)
package pyvenv

import (
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

//go:embed worker.py
var workerPySource []byte

// Sentinel errors returned from Call, Await, Result, Start, and helpers.
var (
	// ErrNotReady is returned when Call is invoked before Start has succeeded.
	ErrNotReady = errors.New("pyvenv: not started")
	// ErrClosed is returned when Call, Await, or Start is invoked after Close.
	ErrClosed = errors.New("pyvenv: closed")
	// ErrDied is returned when the Python subprocess exited unexpectedly.
	ErrDied = errors.New("pyvenv: python subprocess exited unexpectedly")
	// ErrUnknown is returned by Await or Result when the callID is not known to the Venv —
	// either because it was never issued, the result was already consumed by a prior Await,
	// or the result aged out of the cache.
	ErrUnknown = errors.New("pyvenv: unknown callID")
)

// State identifies an async lifecycle transition observed by a [LivenessCallback].
type State int

const (
	// StateReady fires when [Venv.Start] has successfully spawned the worker and the worker
	// has emitted its initial ready frame.
	StateReady State = iota
	// StateDied fires when the subprocess exits unexpectedly. Subsequent [Venv.Call] returns
	// [ErrDied] until [Venv.Start] is called again or [Venv.Close] terminates the Venv.
	StateDied
)

// LivenessCallback is invoked on async lifecycle transitions. The module invokes the callback
// from a goroutine it owns; do not block. A single Venv may invoke its callback multiple times
// across its lifetime (Ready then Died then Ready after a successful Start retry).
type LivenessCallback func(state State, err error)

// Logger is the destination for diagnostic logs (subprocess lifecycle, pip output, worker
// crashes). The minimal surface lets callers wrap any sink (slog, log, a framework's own
// logger) with a small adapter.
type Logger interface {
	LogInfo(ctx context.Context, msg string, kv ...any)
	LogError(ctx context.Context, msg string, kv ...any)
}

// Config configures a [Venv]. Zero values are sensible defaults.
type Config struct {
	// PythonInterpreter is the bootstrap interpreter used once at venv creation
	// (`<interp> -m venv ...`). After bootstrap the venv has its own pinned interpreter.
	// Default: "python3" resolved via $PATH.
	PythonInterpreter string

	// Sources is the list of Python source files to load into the worker, in order.
	// The module concatenates them and sends one `define` frame; the caller decides
	// ordering (typical convention: helpers first, then a main service.py last so it
	// can reference earlier names).
	Sources []string

	// Requirements is the list of pip packages to install at Start time. Empty list
	// skips the pip install pass. Identical to the lines of a requirements.txt; commas
	// and version specifiers in entries are passed through to pip as-is.
	Requirements []string

	// MaxWorkers is the size of the Python ThreadPoolExecutor that serves concurrent
	// Call requests. Default 1. Use higher values only when the Python code is
	// thread-safe and benefits from concurrent dispatch.
	MaxWorkers int

	// BaseDir is the on-disk directory holding the venv tree (`<dir>/venv/`) and the
	// worker output buffers (`<dir>/meta/`). Default: a temp directory created by the
	// module on first Start.
	BaseDir string

	// OutputBufferKB sizes each of the two ring files per stream. Each Venv has
	// 2 * stream-count files (stdout-a, stdout-b, stderr-a, stderr-b). Default 256.
	OutputBufferKB int

	// ResultCacheTTL caps how long a completed-but-not-yet-Awaited call result lingers
	// before being evicted. Default 15 minutes. Set to a negative value to disable
	// eviction entirely (results stay until consumed, the Venv dies, or it is Closed).
	ResultCacheTTL time.Duration

	// Logger receives diagnostic messages. nil → logs to stderr via [log.Default].
	Logger Logger

	// LivenessCallback observes async lifecycle transitions. nil disables the hook.
	LivenessCallback LivenessCallback
}

// Venv owns one long-lived Python subprocess and its on-disk virtualenv. Safe for concurrent
// Call from multiple goroutines.
type Venv struct {
	cfg Config

	startMu sync.Mutex // serializes Start; one Start at a time per Venv.

	mu       sync.Mutex
	state    venvState
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	stdinMu  sync.Mutex
	stdoutRW *ringWriter
	stderrRW *ringWriter

	// workerReadyOnce + workerReadyCh signal the worker's initial "ready" frame.
	// Replaced at the start of each doStart so retries after StateDied work cleanly.
	workerReadyOnce sync.Once
	workerReadyCh   chan struct{}

	// Call tracking. Each Call inserts an entry; completeCall stores the terminal result
	// and closes the entry's done channel; Await/Result observe done and atomically consume.
	pendingMu   sync.Mutex
	calls       map[string]*callEntry
	pendingOp   string
	pendingOpCh chan opResult

	// stopCleanup is closed by Close to terminate the TTL-eviction goroutine. nil before Start.
	stopCleanup chan struct{}

	closed atomic.Bool
}

type venvState int

const (
	stateInit venvState = iota
	stateStarting
	stateReady
	stateDied
	stateClosed
)

type callResult struct {
	ok      bool
	result  json.RawMessage
	errMsg  string
	errType string

	// sentinelErr, if non-nil, is the exact error to return to the Await/Result caller. Set
	// when the Venv tears down callers' entries on Close or Died so the caller can errors.Is
	// against ErrClosed / ErrDied.
	sentinelErr error
}

// callEntry tracks one in-flight or completed Call. done is closed when result has been
// populated (by completeCall, or by a sentinel teardown path); Await/Result block on done.
// A successful Await consumes the entry by removing it from the calls map under pendingMu;
// subsequent Awaits on the same callID return ErrUnknown.
type callEntry struct {
	funcName    string
	done        chan struct{}
	result      callResult // populated before done is closed; read-only afterward
	completedAt time.Time  // when result was stored; used for TTL eviction
}

type opResult struct {
	ok      bool
	errMsg  string
	errType string
}

// New constructs a Venv. The Python subprocess is not spawned; the caller must call [Venv.Start].
func New(cfg Config) (*Venv, error) {
	if cfg.PythonInterpreter == "" {
		cfg.PythonInterpreter = "python3"
	}
	if cfg.MaxWorkers <= 0 {
		cfg.MaxWorkers = 1
	}
	if cfg.OutputBufferKB <= 0 {
		cfg.OutputBufferKB = 256
	}
	if cfg.ResultCacheTTL == 0 {
		cfg.ResultCacheTTL = 15 * time.Minute
	}
	if cfg.BaseDir == "" {
		dir, err := os.MkdirTemp("", "pyvenv-")
		if err != nil {
			return nil, fmt.Errorf("create temp BaseDir: %w", err)
		}
		cfg.BaseDir = dir
	}
	return &Venv{
		cfg:   cfg,
		calls: map[string]*callEntry{},
	}, nil
}

// Start creates the on-disk venv (skipping the create step if one is already present at
// BaseDir/venv), runs `pip install` if [Config.Requirements] is non-empty, spawns the worker,
// loads the embedded sources, and waits for the worker to emit its initial ready frame.
// Returns when the worker is ready or ctx expires. Safe to retry after [StateDied]: the on-disk
// venv is reused, pip install is skipped if the install marker is current, and a fresh worker
// is spawned.
func (v *Venv) Start(ctx context.Context) error {
	// Serialize Start: a second concurrent Start blocks until the first finishes, then
	// observes the resulting state and returns immediately if Ready.
	v.startMu.Lock()
	defer v.startMu.Unlock()

	if v.closed.Load() {
		return ErrClosed
	}
	v.mu.Lock()
	if v.state == stateReady {
		v.mu.Unlock()
		return nil
	}
	v.state = stateStarting
	v.workerReadyOnce = sync.Once{}
	v.workerReadyCh = make(chan struct{})
	v.mu.Unlock()

	err := v.doStart(ctx)
	if err != nil {
		v.mu.Lock()
		v.state = stateInit
		v.mu.Unlock()
	}
	return err
}

// doStart runs the bootstrap → pip install → spawn worker → load sources → wait-ready sequence.
// On success the Venv is in stateReady and the LivenessCallback has fired with StateReady.
func (v *Venv) doStart(ctx context.Context) error {
	venvDir := filepath.Join(v.cfg.BaseDir, "venv")
	metaDir := filepath.Join(v.cfg.BaseDir, "meta")

	// Set up ring writers (idempotent — first Start creates, subsequent reuse).
	v.mu.Lock()
	if v.stdoutRW == nil {
		rw, err := newRingWriter(metaDir, "stdout", v.cfg.OutputBufferKB*1024)
		if err != nil {
			v.mu.Unlock()
			return err
		}
		v.stdoutRW = rw
	}
	if v.stderrRW == nil {
		rw, err := newRingWriter(metaDir, "stderr", v.cfg.OutputBufferKB*1024)
		if err != nil {
			v.mu.Unlock()
			return err
		}
		v.stderrRW = rw
	}
	v.mu.Unlock()

	// Bootstrap the venv if needed.
	if !venvBinaryExists(venvDir) {
		v.logInfo(ctx, "creating venv", "dir", venvDir)
		cmd := exec.CommandContext(ctx, v.cfg.PythonInterpreter, "-m", "venv", venvDir)
		cmd.Stdout = v.stdoutRW
		cmd.Stderr = v.stderrRW
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("create venv: %w", err)
		}
	}

	// Run pip install if Requirements changed since the last successful install.
	if len(v.cfg.Requirements) > 0 {
		markerPath := filepath.Join(metaDir, ".requirements")
		want := strings.Join(v.cfg.Requirements, "\n")
		got, _ := os.ReadFile(markerPath)
		if string(got) != want {
			v.logInfo(ctx, "pip install", "packages", v.cfg.Requirements)
			args := append([]string{"install"}, v.cfg.Requirements...)
			cmd := exec.CommandContext(ctx, pipPath(venvDir), args...)
			cmd.Stdout = v.stdoutRW
			cmd.Stderr = v.stderrRW
			if err := cmd.Run(); err != nil {
				return fmt.Errorf("pip install: %w", err)
			}
			_ = os.WriteFile(markerPath, []byte(want), 0o644)
		}
	}

	// Write worker.py and spawn it.
	workerPath := filepath.Join(metaDir, "worker.py")
	if err := os.WriteFile(workerPath, workerPySource, 0o644); err != nil {
		return fmt.Errorf("write worker.py: %w", err)
	}
	cmd := exec.Command(pythonPath(venvDir),
		workerPath,
		"--max-workers", strconv.Itoa(v.cfg.MaxWorkers),
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start worker: %w", err)
	}

	v.mu.Lock()
	v.cmd = cmd
	v.stdin = stdin
	v.mu.Unlock()

	// Reset call tracking from any prior incarnation.
	v.pendingMu.Lock()
	v.calls = map[string]*callEntry{}
	v.pendingMu.Unlock()

	// Start the TTL-eviction goroutine if eviction is enabled and one isn't already running.
	v.mu.Lock()
	if v.stopCleanup == nil && v.cfg.ResultCacheTTL > 0 {
		v.stopCleanup = make(chan struct{})
		go v.cleanupLoop(v.stopCleanup, v.cfg.ResultCacheTTL)
	}
	v.mu.Unlock()

	// Forward worker stderr to the ring + logger.
	go func() {
		_, _ = io.Copy(v.stderrRW, stderr)
	}()
	go v.readLoop(stdout)
	go v.waitLoop(cmd)

	// Wait for worker's ready frame.
	if err := v.waitReady(ctx); err != nil {
		// Kill the partially-spawned worker so its goroutines exit cleanly.
		_ = v.stdin.Close()
		_ = cmd.Process.Kill()
		return fmt.Errorf("waiting for worker ready: %w", err)
	}

	// Load embedded sources via one Define frame.
	if joined := strings.Join(v.cfg.Sources, "\n"); joined != "" {
		if err := v.define(ctx, joined); err != nil {
			_ = v.stdin.Close()
			_ = cmd.Process.Kill()
			return fmt.Errorf("define sources: %w", err)
		}
	}

	v.mu.Lock()
	v.state = stateReady
	v.mu.Unlock()
	v.fireLiveness(StateReady, nil)
	return nil
}

// readLoop dispatches incoming frames until the worker's stdout closes.
func (v *Venv) readLoop(r io.Reader) {
	_ = readFrames(r, v.dispatch)
}

// waitLoop blocks on the subprocess's exit and transitions to stateDied on unexpected exit.
// The Died callback only fires when the worker had previously reached stateReady - dying
// before Ready is treated as a Start failure and surfaces via Start's return error rather
// than via the liveness callback (no point telling the caller "it died" when they were
// never told "it was ready").
func (v *Venv) waitLoop(cmd *exec.Cmd) {
	waitErr := cmd.Wait()
	v.mu.Lock()
	if v.state == stateClosed || v.state != stateReady {
		v.mu.Unlock()
		return
	}
	v.state = stateDied
	v.mu.Unlock()

	// Mark all pending calls as died. Entries remain in the map so a waiter still parked on
	// Await sees ErrDied via the consume path; a TTL pass eventually reaps any that no one
	// Awaits.
	diedErr := ErrDied
	if waitErr != nil {
		diedErr = fmt.Errorf("%w: %v", ErrDied, waitErr)
	}
	v.pendingMu.Lock()
	for _, entry := range v.calls {
		select {
		case <-entry.done:
			// already terminal; leave the prior result in place
		default:
			entry.result = callResult{sentinelErr: diedErr}
			entry.completedAt = time.Now()
			close(entry.done)
		}
	}
	opCh := v.pendingOpCh
	v.pendingOp = ""
	v.pendingOpCh = nil
	v.pendingMu.Unlock()
	if opCh != nil {
		select {
		case opCh <- opResult{ok: false, errMsg: ErrDied.Error()}:
		default:
		}
		close(opCh)
	}

	v.fireLiveness(StateDied, diedErr)
}

// dispatch routes a single received frame to the appropriate handler.
func (v *Venv) dispatch(frame map[string]any) {
	t, _ := frame["type"].(string)
	switch t {
	case frameReady:
		v.workerReadyOnce.Do(func() {
			close(v.workerReady())
		})
	case frameOpDone:
		opID, _ := frame["opID"].(string)
		ok, _ := frame["ok"].(bool)
		errMsg, _ := frame["errorMessage"].(string)
		errType, _ := frame["errorType"].(string)
		v.completeOp(opID, opResult{ok: ok, errMsg: errMsg, errType: errType})
	case frameCallDone:
		callID, _ := frame["callID"].(string)
		ok, _ := frame["ok"].(bool)
		errMsg, _ := frame["errorMessage"].(string)
		errType, _ := frame["errorType"].(string)
		raw, _ := json.Marshal(frame["result"])
		v.completeCall(callID, callResult{ok: ok, result: raw, errMsg: errMsg, errType: errType})
	case frameError:
		errMsg, _ := frame["errorMessage"].(string)
		errType, _ := frame["errorType"].(string)
		v.logError(context.Background(), "worker reported error",
			"errorType", errType, "errorMessage", errMsg)
	}
}

// workerReady is a getter that captures the ready channel under the mutex so the read inside
// workerReadyOnce.Do sees a consistent value across Start retries.
func (v *Venv) workerReady() chan struct{} {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.workerReadyCh
}

// waitReady blocks until the worker's ready frame arrives or ctx expires.
func (v *Venv) waitReady(ctx context.Context) error {
	select {
	case <-v.workerReady():
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// define sends one define frame carrying the concatenated Python source and waits for the
// matching op_done frame.
func (v *Venv) define(ctx context.Context, code string) error {
	opID := randID()
	ch := make(chan opResult, 1)

	v.pendingMu.Lock()
	v.pendingOp = opID
	v.pendingOpCh = ch
	v.pendingMu.Unlock()

	err := v.writeFrame(map[string]any{
		"type": frameDefine,
		"opID": opID,
		"code": code,
	})
	if err != nil {
		v.pendingMu.Lock()
		v.pendingOp = ""
		v.pendingOpCh = nil
		v.pendingMu.Unlock()
		return err
	}

	select {
	case res := <-ch:
		if !res.ok {
			return fmt.Errorf("define failed: %s: %s", res.errType, res.errMsg)
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// completeOp delivers the result of a pending state op (define) to its waiter.
func (v *Venv) completeOp(opID string, res opResult) {
	v.pendingMu.Lock()
	if v.pendingOp != opID {
		v.pendingMu.Unlock()
		return
	}
	ch := v.pendingOpCh
	v.pendingOp = ""
	v.pendingOpCh = nil
	v.pendingMu.Unlock()
	if ch != nil {
		select {
		case ch <- res:
		default:
		}
		close(ch)
	}
}

// Call asynchronously invokes a named Python function with the given JSON-marshaled args.
// Returns a callID that the caller passes to [Venv.Await] or [Venv.Result] to retrieve the
// result. Call does not block on Python execution; it validates state, registers the call,
// writes one frame, and returns.
//
// The Python work runs to completion in the worker regardless of whether the caller's ctx
// expires before Await returns. A subsequent Await(callID) — including one from a separate
// goroutine or after a workflow-task retry that persisted the callID — re-enters the wait
// and receives the result once Python finishes.
//
// May be called from multiple goroutines concurrently. Each in-flight Call consumes one slot
// in the Python [Config.MaxWorkers] thread pool; additional calls queue inside the worker.
func (v *Venv) Call(ctx context.Context, funcName string, args any) (callID string, err error) {
	if v.closed.Load() {
		return "", ErrClosed
	}
	v.mu.Lock()
	state := v.state
	v.mu.Unlock()
	switch state {
	case stateInit, stateStarting:
		return "", ErrNotReady
	case stateDied:
		return "", ErrDied
	case stateClosed:
		return "", ErrClosed
	}

	callID = randID()
	entry := &callEntry{
		funcName: funcName,
		done:     make(chan struct{}),
	}
	v.pendingMu.Lock()
	v.calls[callID] = entry
	v.pendingMu.Unlock()

	err = v.writeFrame(map[string]any{
		"type":   frameCall,
		"callID": callID,
		"func":   funcName,
		"args":   args,
	})
	if err != nil {
		v.pendingMu.Lock()
		delete(v.calls, callID)
		v.pendingMu.Unlock()
		return "", err
	}
	return callID, nil
}

// Await blocks until the call identified by callID completes or ctx expires. On successful
// delivery the result is unmarshaled into *result and the entry is consumed — subsequent
// Awaits on the same callID return [ErrUnknown]. On Python failure, Await returns an error
// whose message includes the Python exception type and message and the entry is consumed.
// On ctx expiry, Await returns ctx.Err() without consuming the entry; a later Await may
// re-enter the wait.
//
// Returns [ErrUnknown] when callID is not known to this Venv (never issued, already
// consumed, or evicted by the result cache's TTL).
//
// Pass result = nil when the caller does not care about the unmarshaled value.
func (v *Venv) Await(ctx context.Context, callID string, result any) error {
	v.pendingMu.Lock()
	entry, ok := v.calls[callID]
	v.pendingMu.Unlock()
	if !ok {
		return ErrUnknown
	}

	select {
	case <-entry.done:
		return v.consume(callID, entry, result)
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Result is the non-blocking sibling of [Venv.Await]. If a terminal result is available it is
// consumed and unmarshaled into *result (or returned via err when Python raised). If the call
// is still running, returns ready=false, err=nil without consuming. If callID is unknown,
// returns ready=false, err=[ErrUnknown].
//
// As with Await, a successful delivery (ready=true) consumes the entry; subsequent Result or
// Await on the same callID returns [ErrUnknown].
func (v *Venv) Result(ctx context.Context, callID string, result any) (ready bool, err error) {
	v.pendingMu.Lock()
	entry, ok := v.calls[callID]
	v.pendingMu.Unlock()
	if !ok {
		return false, ErrUnknown
	}

	select {
	case <-entry.done:
		return true, v.consume(callID, entry, result)
	default:
		return false, nil
	}
}

// CallAndAwait is the synchronous convenience shape: Call followed by Await on the same
// goroutine. Use it when the caller is happy to block for the duration of the Python work
// and does not need to hand the callID across a context boundary (e.g. a workflow task
// retry, or a separate progress-tracking goroutine).
func (v *Venv) CallAndAwait(ctx context.Context, funcName string, args, result any) error {
	callID, err := v.Call(ctx, funcName, args)
	if err != nil {
		return err
	}
	return v.Await(ctx, callID, result)
}

// consume atomically removes the entry from the calls map and translates its result into an
// Await/Result return value. The atomic check-then-delete ensures that two concurrent Awaits
// on the same callID deliver to exactly one caller; the other sees the entry gone and
// returns [ErrUnknown]. (The framework does not normally create concurrent Awaits — workflow
// retries are sequential and Fork starts from a committed checkpoint — but the property
// holds for any caller that constructs the race deliberately.)
func (v *Venv) consume(callID string, entry *callEntry, result any) error {
	v.pendingMu.Lock()
	_, stillThere := v.calls[callID]
	if !stillThere {
		v.pendingMu.Unlock()
		return ErrUnknown
	}
	delete(v.calls, callID)
	v.pendingMu.Unlock()

	res := entry.result
	if res.sentinelErr != nil {
		return res.sentinelErr
	}
	if !res.ok {
		return fmt.Errorf("python call %s failed: %s: %s", entry.funcName, res.errType, res.errMsg)
	}
	if result == nil {
		return nil
	}
	return json.Unmarshal(res.result, result)
}

// completeCall is invoked by the dispatch goroutine when a call_done frame arrives. It stores
// the terminal result on the entry and signals waiters by closing done. The entry is not
// removed here; consumption happens in [Venv.Await] / [Venv.Result] so that a caller that
// hasn't reached its Await yet can still observe the result, and TTL eviction has visibility
// to age out orphan results.
func (v *Venv) completeCall(callID string, res callResult) {
	v.pendingMu.Lock()
	entry, ok := v.calls[callID]
	v.pendingMu.Unlock()
	if !ok {
		return
	}
	entry.result = res
	entry.completedAt = time.Now()
	close(entry.done)
}

// cleanupLoop is the background TTL evictor. It scans completed-but-not-consumed entries
// once per ttl/2 (so an entry's worst-case lingering time is ttl + ttl/2) and drops any
// whose completedAt + ttl is in the past. Stops when stopCleanup is closed by [Venv.Close].
func (v *Venv) cleanupLoop(stop <-chan struct{}, ttl time.Duration) {
	interval := ttl / 2
	if interval < time.Second {
		interval = time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case now := <-t.C:
			cutoff := now.Add(-ttl)
			v.pendingMu.Lock()
			for callID, entry := range v.calls {
				select {
				case <-entry.done:
					if entry.completedAt.Before(cutoff) {
						delete(v.calls, callID)
					}
				default:
				}
			}
			v.pendingMu.Unlock()
		}
	}
}

// writeFrame serializes a frame on the subprocess's stdin under stdinMu so concurrent writes
// don't interleave their header+body pairs on the wire.
func (v *Venv) writeFrame(obj map[string]any) error {
	v.stdinMu.Lock()
	defer v.stdinMu.Unlock()
	if v.stdin == nil {
		return ErrNotReady
	}
	return writeFrame(v.stdin, obj)
}

// Ready reports whether the worker is up and accepting Call.
func (v *Venv) Ready() bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.state == stateReady
}

// TailStdOut returns the most recent stdout from the worker, up to 2 * OutputBufferKB bytes.
// Pull-based; cheap. Returns nil before Start and after Close.
func (v *Venv) TailStdOut() []byte {
	v.mu.Lock()
	rw := v.stdoutRW
	v.mu.Unlock()
	if rw == nil {
		return nil
	}
	return rw.Tail()
}

// TailStdErr returns the most recent stderr from the worker.
func (v *Venv) TailStdErr() []byte {
	v.mu.Lock()
	rw := v.stderrRW
	v.mu.Unlock()
	if rw == nil {
		return nil
	}
	return rw.Tail()
}

// Close kills the subprocess, closes the ring writers, removes the on-disk venv directory, and
// marks the Venv terminal. In-flight Await returns [ErrClosed]. Idempotent.
func (v *Venv) Close(ctx context.Context) error {
	if !v.closed.CompareAndSwap(false, true) {
		return nil
	}
	v.mu.Lock()
	v.state = stateClosed
	cmd := v.cmd
	stdin := v.stdin
	stdoutRW := v.stdoutRW
	stderrRW := v.stderrRW
	stopCleanup := v.stopCleanup
	v.cmd = nil
	v.stdin = nil
	v.stdoutRW = nil
	v.stderrRW = nil
	v.stopCleanup = nil
	baseDir := v.cfg.BaseDir
	v.mu.Unlock()

	if stopCleanup != nil {
		close(stopCleanup)
	}
	if stdin != nil {
		_ = stdin.Close()
	}
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
	if stdoutRW != nil {
		_ = stdoutRW.Close()
	}
	if stderrRW != nil {
		_ = stderrRW.Close()
	}

	// Tear down any pending callers: closing each entry's done with sentinelErr=ErrClosed
	// wakes parked Awaits, which then read the sentinel via consume() and return ErrClosed.
	// Entries are left in the map so the atomic check-then-delete in consume() finds them
	// and routes through sentinelErr; the Venv itself is about to be unreferenced, so GC
	// reclaims any orphan entries shortly after.
	v.pendingMu.Lock()
	for _, entry := range v.calls {
		select {
		case <-entry.done:
			// already terminal
		default:
			entry.result = callResult{sentinelErr: ErrClosed}
			entry.completedAt = time.Now()
			close(entry.done)
		}
	}
	v.pendingMu.Unlock()

	if baseDir != "" {
		_ = os.RemoveAll(baseDir)
	}
	return nil
}

// fireLiveness invokes the LivenessCallback off the caller's goroutine to avoid blocking the
// dispatch / wait loops on slow user code. The callback documentation states "do not block,"
// but we still isolate it via a goroutine as defense in depth.
func (v *Venv) fireLiveness(state State, err error) {
	cb := v.cfg.LivenessCallback
	if cb == nil {
		return
	}
	go cb(state, err)
}

func (v *Venv) logInfo(ctx context.Context, msg string, kv ...any) {
	if v.cfg.Logger != nil {
		v.cfg.Logger.LogInfo(ctx, msg, kv...)
	}
}

func (v *Venv) logError(ctx context.Context, msg string, kv ...any) {
	if v.cfg.Logger != nil {
		v.cfg.Logger.LogError(ctx, msg, kv...)
	}
}

// --- helpers ---

func pythonPath(venvDir string) string {
	return filepath.Join(venvDir, "bin", "python")
}

func pipPath(venvDir string) string {
	return filepath.Join(venvDir, "bin", "pip")
}

func venvBinaryExists(venvDir string) bool {
	_, err := os.Stat(pythonPath(venvDir))
	return err == nil
}

func randID() string {
	var buf [16]byte
	_, _ = rand.Read(buf[:])
	return hex.EncodeToString(buf[:])
}

