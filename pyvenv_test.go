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

package pyvenv

import (
	"context"
	"errors"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

// hasPython3 returns true when a `python3 -m venv --help` invocation succeeds. Tests that
// need a real Python skip when this returns false, so the suite stays green on minimal
// images (e.g. debian-slim without python3-venv).
func hasPython3(t *testing.T) bool {
	t.Helper()
	cmd := exec.Command("python3", "-m", "venv", "--help")
	if err := cmd.Run(); err != nil {
		t.Skip("python3 venv not available:", err)
		return false
	}
	return true
}

// TestNew_Defaults asserts the zero-value Config fills in sensible defaults and the Venv is
// constructible without spawning anything.
func TestNew_Defaults(t *testing.T) {
	t.Parallel()
	v, err := New(Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	if v.cfg.PythonInterpreter != "python3" {
		t.Errorf("PythonInterpreter default = %q, want python3", v.cfg.PythonInterpreter)
	}
	if v.cfg.MaxWorkers != 1 {
		t.Errorf("MaxWorkers default = %d, want 1", v.cfg.MaxWorkers)
	}
	if v.cfg.OutputBufferKB != 256 {
		t.Errorf("OutputBufferKB default = %d, want 256", v.cfg.OutputBufferKB)
	}
	if v.cfg.ResultCacheTTL != 15*time.Minute {
		t.Errorf("ResultCacheTTL default = %v, want 15m", v.cfg.ResultCacheTTL)
	}
	if v.cfg.BaseDir == "" {
		t.Error("BaseDir default = empty, want a temp dir")
	}
}

// TestCall_NotReady asserts Call returns ErrNotReady before Start.
func TestCall_NotReady(t *testing.T) {
	t.Parallel()
	v, _ := New(Config{})
	defer v.Close(context.Background())
	_, err := v.Call(context.Background(), "anyfunc", nil)
	if !errors.Is(err, ErrNotReady) {
		t.Errorf("Call before Start: got %v, want ErrNotReady", err)
	}
}

// TestCall_AfterClose asserts Call returns ErrClosed after Close.
func TestCall_AfterClose(t *testing.T) {
	t.Parallel()
	v, _ := New(Config{})
	_ = v.Close(context.Background())
	_, err := v.Call(context.Background(), "anyfunc", nil)
	if !errors.Is(err, ErrClosed) {
		t.Errorf("Call after Close: got %v, want ErrClosed", err)
	}
}

// TestAwait_Unknown asserts Await on a never-issued callID returns ErrUnknown.
func TestAwait_Unknown(t *testing.T) {
	t.Parallel()
	v, _ := New(Config{})
	defer v.Close(context.Background())
	err := v.Await(context.Background(), "no-such-id", nil)
	if !errors.Is(err, ErrUnknown) {
		t.Errorf("Await on unknown callID: got %v, want ErrUnknown", err)
	}
}

// TestResult_Unknown asserts Result on a never-issued callID returns (false, ErrUnknown).
func TestResult_Unknown(t *testing.T) {
	t.Parallel()
	v, _ := New(Config{})
	defer v.Close(context.Background())
	ready, err := v.Result(context.Background(), "no-such-id", nil)
	if ready {
		t.Error("Result on unknown callID: ready=true, want false")
	}
	if !errors.Is(err, ErrUnknown) {
		t.Errorf("Result on unknown callID: got %v, want ErrUnknown", err)
	}
}

// TestClose_Idempotent asserts a second Close is a no-op.
func TestClose_Idempotent(t *testing.T) {
	t.Parallel()
	v, _ := New(Config{})
	if err := v.Close(context.Background()); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := v.Close(context.Background()); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// TestCallAndAwait_NoRequirements covers the happy path with no pip install: spawn a venv,
// load a trivial function, CallAndAwait it, get the result back.
func TestCallAndAwait_NoRequirements(t *testing.T) {
	// No parallel - spawns a real Python venv and bootstraps it on disk.
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`def echo(args):
    return {"echoed": args.get("text", "")}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if !v.Ready() {
		t.Fatal("Ready() = false after successful Start")
	}

	var out struct {
		Echoed string `json:"echoed"`
	}
	err = v.CallAndAwait(ctx, "echo", map[string]any{"text": "hello"}, &out)
	if err != nil {
		t.Fatalf("CallAndAwait: %v", err)
	}
	if out.Echoed != "hello" {
		t.Errorf("CallAndAwait result = %q, want %q", out.Echoed, "hello")
	}
}

// TestCall_Await_Split exercises the split API: Call returns a callID, a separate goroutine
// (or the same one) then Awaits using that callID.
func TestCall_Await_Split(t *testing.T) {
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`def echo(args):
    return {"echoed": args.get("text", "")}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	callID, err := v.Call(ctx, "echo", map[string]any{"text": "split"})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if callID == "" {
		t.Fatal("Call: empty callID")
	}

	var out struct {
		Echoed string `json:"echoed"`
	}
	if err := v.Await(ctx, callID, &out); err != nil {
		t.Fatalf("Await: %v", err)
	}
	if out.Echoed != "split" {
		t.Errorf("Await result = %q, want %q", out.Echoed, "split")
	}

	// A second Await on the same callID returns ErrUnknown — single-delivery semantics.
	if err := v.Await(ctx, callID, &out); !errors.Is(err, ErrUnknown) {
		t.Errorf("second Await: got %v, want ErrUnknown", err)
	}
}

// TestAwait_CtxExpiryDoesNotConsume verifies that ctx-expiry on Await leaves the in-flight
// call resumable: a subsequent Await on the same callID picks up the result once Python
// finishes.
func TestAwait_CtxExpiryDoesNotConsume(t *testing.T) {
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`import time
def slow(args):
    time.sleep(args.get("seconds", 1))
    return {"done": True}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	startCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(startCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	callCtx, callCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer callCancel()
	callID, err := v.Call(callCtx, "slow", map[string]any{"seconds": 1})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	// First Await with a deadline shorter than the Python sleep — should return ctx error
	// without consuming the entry.
	shortCtx, shortCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer shortCancel()
	err = v.Await(shortCtx, callID, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("first Await: got %v, want DeadlineExceeded", err)
	}

	// Second Await with a longer deadline — should pick up the result the Python function
	// eventually produces.
	longCtx, longCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer longCancel()
	var out struct {
		Done bool `json:"done"`
	}
	if err := v.Await(longCtx, callID, &out); err != nil {
		t.Fatalf("second Await: %v", err)
	}
	if !out.Done {
		t.Errorf("second Await result = %+v, want Done=true", out)
	}
}

// TestAwait_MultipleTimeoutsThenSuccess walks the durability pattern: one Call into a Python
// function that sleeps for 2s, three Awaits with an 800ms deadline each. The first two
// timeouts must return DeadlineExceeded without consuming the entry. The third must return
// the eventual result. This is the workflow-task pattern the library was redesigned for:
// the workflow retries the task, each retry re-enters Await on the same persisted callID.
func TestAwait_MultipleTimeoutsThenSuccess(t *testing.T) {
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`import time
def slowval(args):
    time.sleep(2)
    return {"v": 42}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	startCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(startCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	callID, err := v.Call(context.Background(), "slowval", nil)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	// First two Awaits: each capped at 800ms, both should time out.
	for i := 1; i <= 2; i++ {
		ctx, cancelAwait := context.WithTimeout(context.Background(), 800*time.Millisecond)
		err = v.Await(ctx, callID, nil)
		cancelAwait()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Await #%d: got %v, want DeadlineExceeded", i, err)
		}
	}

	// Third Await: the Python sleep is roughly done by now; the cap of 800ms still gives
	// it plenty of headroom for the result frame to round-trip back.
	ctx3, cancel3 := context.WithTimeout(context.Background(), 800*time.Millisecond)
	defer cancel3()
	var out struct {
		V int `json:"v"`
	}
	if err := v.Await(ctx3, callID, &out); err != nil {
		t.Fatalf("Await #3: %v", err)
	}
	if out.V != 42 {
		t.Errorf("Await #3 result = %d, want 42", out.V)
	}

	// The result was consumed; a fourth Await returns ErrUnknown.
	if err := v.Await(context.Background(), callID, &out); !errors.Is(err, ErrUnknown) {
		t.Errorf("Await #4 (after consume): got %v, want ErrUnknown", err)
	}
}

// TestResult_NonBlocking verifies Result returns (false, nil) for a still-running call and
// (true, nil) with the result populated once the call completes.
func TestResult_NonBlocking(t *testing.T) {
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`import time
def slow(args):
    time.sleep(args.get("seconds", 1))
    return {"done": True}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	callID, err := v.Call(ctx, "slow", map[string]any{"seconds": 1})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	// Immediately after Call: not ready yet.
	ready, err := v.Result(ctx, callID, nil)
	if err != nil {
		t.Errorf("Result while pending: err=%v, want nil", err)
	}
	if ready {
		t.Error("Result while pending: ready=true, want false")
	}

	// Wait it out via Await.
	if err := v.Await(ctx, callID, nil); err != nil {
		t.Fatalf("Await: %v", err)
	}

	// After the result was consumed by Await, Result returns ErrUnknown.
	ready, err = v.Result(ctx, callID, nil)
	if ready {
		t.Error("Result after Await consumed: ready=true, want false")
	}
	if !errors.Is(err, ErrUnknown) {
		t.Errorf("Result after Await consumed: got %v, want ErrUnknown", err)
	}
}

// TestResult_AfterCompletion verifies Result returns (true, nil) and populates the result
// when the Python call has already finished.
func TestResult_AfterCompletion(t *testing.T) {
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`def double(args):
    return {"v": args["x"] * 2}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	callID, err := v.Call(ctx, "double", map[string]any{"x": 21})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	// Poll Result until ready. The double() call is sub-millisecond on a working host but
	// we still need to let the worker dispatch + the response frame round-trip.
	deadline := time.Now().Add(5 * time.Second)
	var out struct {
		V int `json:"v"`
	}
	for {
		ready, err := v.Result(ctx, callID, &out)
		if err != nil {
			t.Fatalf("Result: %v", err)
		}
		if ready {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("Result never reported ready within 5s")
		}
		time.Sleep(20 * time.Millisecond)
	}
	if out.V != 42 {
		t.Errorf("Result value = %d, want 42", out.V)
	}
}

// TestLivenessCallback_Ready asserts the callback fires StateReady after a successful Start.
func TestLivenessCallback_Ready(t *testing.T) {
	// No parallel - spawns a real Python venv.
	if !hasPython3(t) {
		return
	}
	var (
		mu     sync.Mutex
		states []State
		done   = make(chan struct{}, 4)
	)
	v, err := New(Config{
		Sources: []string{`def noop(args): return None
`},
		LivenessCallback: func(state State, err error) {
			mu.Lock()
			states = append(states, state)
			mu.Unlock()
			done <- struct{}{}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for LivenessCallback")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(states) == 0 || states[0] != StateReady {
		t.Errorf("first liveness transition = %v, want StateReady", states)
	}
}

// TestLivenessCallback_Died asserts the callback fires StateDied when the subprocess exits
// unexpectedly (we kill it ourselves to simulate a crash).
func TestLivenessCallback_Died(t *testing.T) {
	// No parallel - spawns a real Python venv and kills it.
	if !hasPython3(t) {
		return
	}
	var (
		mu    sync.Mutex
		seen  []State
		ready = make(chan struct{})
		died  = make(chan struct{})
	)
	v, err := New(Config{
		Sources: []string{`def noop(args): return None
`},
		LivenessCallback: func(state State, err error) {
			mu.Lock()
			seen = append(seen, state)
			mu.Unlock()
			switch state {
			case StateReady:
				select {
				case ready <- struct{}{}:
				default:
				}
			case StateDied:
				select {
				case died <- struct{}{}:
				default:
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	select {
	case <-ready:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for StateReady")
	}

	// Kill the worker directly to simulate a crash.
	v.mu.Lock()
	cmd := v.cmd
	v.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		t.Fatal("no worker process to kill")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("Kill: %v", err)
	}

	select {
	case <-died:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for StateDied")
	}

	// A Call after Died should return ErrDied.
	_, err = v.Call(context.Background(), "noop", nil)
	if !errors.Is(err, ErrDied) {
		t.Errorf("Call after Died: got %v, want ErrDied", err)
	}
}

// TestAwait_AfterDeath asserts that an Await parked on an in-flight call wakes with ErrDied
// when the subprocess crashes.
func TestAwait_AfterDeath(t *testing.T) {
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`import time
def slow(args):
    time.sleep(5)
    return {"done": True}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	callID, err := v.Call(ctx, "slow", nil)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	awaitErr := make(chan error, 1)
	go func() {
		awaitErr <- v.Await(context.Background(), callID, nil)
	}()

	// Give Await a moment to park.
	time.Sleep(100 * time.Millisecond)

	v.mu.Lock()
	cmd := v.cmd
	v.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}

	select {
	case err := <-awaitErr:
		if !errors.Is(err, ErrDied) {
			t.Errorf("Await after Died: got %v, want ErrDied", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Await did not wake after subprocess death")
	}
}

// TestStart_Idempotent asserts that two concurrent Start calls converge on the same readiness
// rather than each spawning a worker.
func TestStart_Idempotent(t *testing.T) {
	// No parallel - spawns a real Python venv.
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{`def noop(args): return None
`},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); _ = v.Start(ctx) }()
	go func() { defer wg.Done(); _ = v.Start(ctx) }()
	wg.Wait()

	if !v.Ready() {
		t.Fatal("Ready() = false after concurrent Start")
	}
}

// TestTail_PreStart_NilSafe asserts that calling Tail* before Start returns nil rather than
// panicking. Lets callers stash a *Venv on a service struct unconditionally and surface
// diagnostics even when Start hasn't run yet.
func TestTail_PreStart_NilSafe(t *testing.T) {
	t.Parallel()
	v, _ := New(Config{})
	defer v.Close(context.Background())
	if out := v.TailStdOut(); out != nil {
		t.Errorf("TailStdOut before Start = %q, want nil", out)
	}
	if out := v.TailStdErr(); out != nil {
		t.Errorf("TailStdErr before Start = %q, want nil", out)
	}
}

// TestSources_Concatenation_Order asserts the caller controls ordering by passing helpers
// before service.py-equivalent definitions.
func TestSources_Concatenation_Order(t *testing.T) {
	// No parallel - spawns a real Python venv.
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{
			`MULTIPLIER = 3
`,
			// Second source refers to the constant from the first.
			`def triple(args):
    return {"value": args["x"] * MULTIPLIER}
`,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	var out struct {
		Value int `json:"value"`
	}
	if err := v.CallAndAwait(ctx, "triple", map[string]any{"x": 7}, &out); err != nil {
		t.Fatalf("CallAndAwait: %v", err)
	}
	if out.Value != 21 {
		t.Errorf("triple(7) = %d, want 21", out.Value)
	}
}

// TestCall_PythonError surfaces the Python exception type and message through Await's error.
func TestCall_PythonError(t *testing.T) {
	// No parallel - spawns a real Python venv.
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		Sources: []string{`def boom(args):
    raise ValueError("expected")
`},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	err = v.CallAndAwait(ctx, "boom", nil, nil)
	if err == nil {
		t.Fatal("CallAndAwait: expected error from raising function, got nil")
	}
	if !strings.Contains(err.Error(), "ValueError") || !strings.Contains(err.Error(), "expected") {
		t.Errorf("error = %q, expected to mention ValueError and 'expected'", err)
	}
}

// TestCall_Concurrent dispatches multiple Calls in parallel into a venv with MaxWorkers=4
// and asserts they all complete with correct results. Exercises the calls map and stdin
// write serialization under contention.
func TestCall_Concurrent(t *testing.T) {
	// No parallel - spawns a real Python venv.
	if !hasPython3(t) {
		return
	}
	v, err := New(Config{
		MaxWorkers: 4,
		Sources: []string{`def square(args):
    return {"v": args["x"] * args["x"]}
`},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := v.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	const N = 16
	var wg sync.WaitGroup
	wg.Add(N)
	errs := make(chan error, N)
	for i := 0; i < N; i++ {
		x := i
		go func() {
			defer wg.Done()
			var out struct {
				V int `json:"v"`
			}
			if err := v.CallAndAwait(ctx, "square", map[string]any{"x": x}, &out); err != nil {
				errs <- err
				return
			}
			if out.V != x*x {
				errs <- errors.New("wrong result")
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent Call failed: %v", err)
	}
}
