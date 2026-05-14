# pyvenv

Runs a Python virtual environment as an in-process worker for a Go program. The Go side spawns
one long-lived Python subprocess per `Venv`, talks to it over length-prefixed JSON frames on
stdin/stdout, and exposes typed `Call` into named Python functions.

The [README](README.md) covers how to use the package. This file captures *why* it is shaped
the way it is. The big design choices and the gotchas a future maintainer will hit.

## Design Rationale

### Why a Go package, not a long-running RPC service

An obvious alternative shape was a standalone process that hosts venvs and serves them over
RPC (gRPC, HTTP, NATS) to many client programs. That shape carries real cost — a separate
deployment artifact, IDs that flow over the wire, cross-instance forwarding when a request
lands on the wrong host, capacity-aware allocation, idle eviction, heartbeats, and a result
store keyed by call ID.

In practice, the relationship between a Go program and its Python venv is 1:1. The Go program
already owns a process, a config, a lifecycle; the venv is just compute it needs to call into.
Bus indirection buys nothing the canonical pattern uses — it just adds latency and operational
surface.

Collapsing the bridge into an in-process Go package strips out everything that existed to
serve the multi-process case: no IDs, no forwarding, no allocation pool, no idle eviction.
The Go program owns its Python subprocess the same way it owns its SQL connection. Faster
(local pipe, no network round-trip per call), simpler (`Call(ctx, fn, args)` returns a
callID; `Await(ctx, callID, &out)` blocks for the result), and the operational surface drops
from a service-with-its-own-replicas to a library import.

The cost we accept: zero cross-program venv sharing. Two Go programs using `sentence-
transformers` each pip-install it into their own venv. Caching that across programs is an
orchestration concern (shared volume, pre-built image) and explicitly out of scope.

### Why stdin/stdout JSON frames

The wire is length-prefixed JSON over the worker's stdin/stdout: 4-byte big-endian length,
then a JSON body. The Python side mirrors the same encoding (see `worker.py`'s `write_frame`/
`read_frame`).

Alternatives considered: Unix sockets, gRPC, shared memory.

- **Unix sockets**: portable enough, but require coordinating socket paths, cleanup on
  unexpected termination, permissions, and an extra connection handshake. stdin/stdout gives
  the same byte stream guarantees without any path or lifecycle to manage.
- **gRPC**: adds a non-trivial Python and Go dependency for a one-process-deep RPC. Overkill
  for "call this function and wait."
- **Shared memory**: fastest, but the framing problem just moves — you still need an
  IPC-grade ring buffer or queue, plus all the cleanup-on-crash machinery that the kernel
  hands you for free with pipes.

stdin/stdout is the cheapest correct choice. The kernel cleans up pipes when either side dies.
No paths, no permissions, no extra deps. JSON keeps the protocol introspectable — `cat`-ing
the bytes between Go and Python during debugging is human-readable.

### Why worker.py is embedded, not pip-installed

`worker.py` is bundled into the Go binary via `//go:embed worker.py`. Start writes it to disk
at `BaseDir/meta/worker.py` and runs it. The alternative — installing a `pyvenv-worker`
package via pip — would couple the Python-side script to the published version of a separate
PyPI artifact, which would have to be kept in sync with the Go module.

Embedding means the Go binary and the worker that runs in the venv are atomically the same
version. Bumping the Go module bumps both halves of the protocol. No semver coordination, no
"pip install failed because the worker package wasn't on the mirror" failure mode.

The size cost is ~3 KB embedded; negligible.

### Why `Start` is serialized via a mutex

`startMu` wraps the entire `Start` method. A concurrent second `Start` blocks until the first
returns, then observes `stateReady` and returns immediately.

The earlier design used a `readyCh` that was both:

1. Closed by `dispatch` when the worker emitted its `ready` frame, and
2. Closed by `Start` when `Start` finished loading sources.

The first close had to come before Define ran (it was what unblocked Define), and the second
had to come after (to let other waiters know Start was actually finished). Putting both onto
one channel ended in a double-close panic the first time Start ran successfully.

Two channels would have worked, but a mutex is simpler: Start either runs or waits for a
prior Start to finish, and the channel `workerReadyCh` is left to do exactly one thing —
signal the worker's `ready` frame. The state field tells callers what happened. No subtle
channel ownership.

The cost: a goroutine calling `Start` while another is mid-`Start` blocks until the first
finishes. That's the right behavior — there's no useful work for two concurrent Start calls
to do anyway.

### LivenessCallback is asymmetric on purpose

The callback fires for two async transitions only: `StateReady` (after a successful Start)
and `StateDied` (when the subprocess exits without a `Close`). It does **not** fire when:

- `Start` returns an error before reaching ready. The caller already received the error
  synchronously; no value in re-delivering it via callback.
- `Close` terminates the subprocess. The caller initiated it; no surprise to report.

The rule is enforced in `waitLoop`: it checks `state != stateReady` before firing `StateDied`.
A subprocess that dies during Start (e.g. ctx timeout before the ready frame arrives) is
treated as a Start failure, not as an unexpected death. Whichever path runs first — Start
returning an error, or waitLoop observing the exit — wins; the other path sees a non-Ready
state and stays quiet.

This keeps the contract clean: "if you got `StateReady`, you may later get `StateDied`. If
you didn't, you won't." The microservice consuming this callback can write a simple
sub-activation / recovery loop without bookkeeping for the never-ready case.

### The pip-install marker is a literal string, not a hash

Start writes `BaseDir/meta/.requirements` containing `strings.Join(Requirements, "\n")`.
Subsequent Starts compare current Requirements against this file; if unchanged, pip install
is skipped.

A SHA256 (or any hash) would work too, but adds nothing — pip cares about the order of
arguments to `pip install`, so any reorder is a "real" change. The literal joined string is
identical-on-the-byte-for-byte basis to "what we'd hand pip," and it's debuggable: `cat
.requirements` shows you exactly what's installed.

Hash would also obscure the diagnostic: an operator looking at the marker can read the
package list directly.

### State machine, minus the state-op error sticky state

The states are `init` → `starting` → `ready`, with `died` as the async failure transition and
`closed` as the terminal state. The earlier `coreservices/venvpy` design had an extra
`state_error` that captured a failed PipInstall or Define and stayed sticky until the caller
explicitly retried or Deallocated.

This module collapses pip install and Define into Start. Either Start succeeds (transition
to `ready`) or it returns an error (state stays `init` so a subsequent Start can retry).
There's no separate "the prior state op failed and the venv is half-mutated" condition to
track because we don't expose Define as a separate operation. The caller can't end up in a
state where the worker is alive but the namespace is broken — Start either gets to ready or
it kills the worker on the way out.

### `Sources []string`, not `fs.FS`

The earlier `coreservices/venvpy` discovered Python sources by walking a directory and
applying a "service.py last, others alphabetical" convention. This module takes a flat
`[]string` of source bodies and concatenates them in slice order.

The convention belongs in the consumer, not the module. A microservice template can walk an
embed.FS and order however it wants — typically helpers first, then a main file last — but
that's a policy decision that varies per project. Embedding it in the module would force one
convention on every consumer.

The cost: callers do ~5 lines of file-reading boilerplate. Acceptable.

### Errors from Python: type + message + traceback

Each `call_done` frame with `ok: false` carries `errorType` (the Python exception class
name), `errorMessage` (its `str()`), and `traceback` (the full formatted traceback). The Go
side surfaces `errorType` and `errorMessage` in the returned error string. The traceback is
written to the stderr ring buffer for diagnostics but not in the error string itself —
tracebacks are typically 10-20 lines of file paths that callers can't easily act on.

Programmatic discrimination ("was this a ValueError or a KeyError?") works via
`strings.Contains(err.Error(), "ValueError")`. Not type-safe, but matches the actual cross-
language interface — Python types don't map cleanly to Go types, so a string predicate is the
honest tool.

### Why the ring buffer is on-disk

`stdout` and `stderr` from the worker are piped through `ringWriter` to two files per stream
(`stdout-a`, `stdout-b`, etc.). `TailStdOut`/`TailStdErr` returns the contents of both files
concatenated.

An in-memory ring buffer would work too. The on-disk version costs nothing extra during
healthy operation (the OS keeps the recent writes in the page cache anyway), and survives an
in-process crash dump or a `kill -SEGV` to the Go program — the worker's last output is
sitting on disk for forensics. With an in-memory buffer the output dies with the process.

The cost: a few file handles per Venv. Trivial.

### Concurrent Call safety

`Call` is safe from multiple goroutines. The serialization points:

- **`pendingMu`** guards the `pendingCalls` map. Insertion (before write) and deletion
  (after receive or ctx-done) happen under the lock.
- **`stdinMu`** wraps each frame write so the (4-byte header, body) pair stays atomic on the
  wire. Without it, two concurrent writes would interleave their headers and bodies and the
  Python side would see corrupted frames.
- **`mu`** guards the state field and the subprocess handles (`cmd`, `stdin`, `stdoutRW`,
  `stderrRW`).

The Python side serves the parallelism via `ThreadPoolExecutor(max_workers=N)`. Caller-side
Calls dispatch to the executor and return via the `call_done` frame's callID — no
serialization in Python beyond what the user's function itself imposes.

### Zero external dependencies

`go.mod` has only the standard library. This is a deliberate constraint:

- The module is consumable from any Go project without dragging in a transitive web.
- Vendoring is trivial.
- An auditor can read the dependency closure in one minute.

The `Logger` interface is defined inside this module rather than imported from any
framework's logging package, precisely to keep the module dependency-free. Consumers wrap
whatever sink they have (slog, log, a framework's own logger) with a small adapter.

### No `Cancel(callID)`

There is no way to interrupt a running Python call:

- Running threads cannot be safely interrupted from outside Python. No portable mechanism
  reaches inside native calls (PyTorch, NumPy hold the GIL until they return), and
  `PyThreadState_SetAsyncExc` is async-unsafe.
- We could `kill -9` the worker, but that loses every other in-flight call too and forces
  a venv rebuild. Not worth it without a concrete use case.

Callers that genuinely need mid-call cancellation should write their Python function to
accept a cooperative cancellation token (a value in `args` they check between safe points).
The module provides no built-in token.

## Things that would surprise future-me

- **`BaseDir` defaults to a `MkdirTemp` directory created by `New`**, not by `Start`. The
  default temp dir is created eagerly so `Close` always has something to clean up. A caller
  that wants persistent venvs (e.g. for fast reuse across program restarts) must set
  `BaseDir` explicitly.

- **`venv` binary existence check skips bootstrap.** Start checks `BaseDir/venv/bin/python`
  and skips `python3 -m venv` if it exists. A partially-created venv (interrupted bootstrap)
  with a missing python binary will re-bootstrap; an interrupted-after-pip venv will skip
  bootstrap, re-run pip (because the requirements marker is stale), and re-spawn the worker.
  Mostly correct; pathological cases require a manual `rm -rf BaseDir`.

- **`workerReadyOnce` is reset on each Start.** Without the reset, a Start-after-Died would
  see the `Do` already fired (from the previous worker's ready frame) and skip the channel
  close, leaving the new doStart blocked on `waitReady` forever. The reset is safe because
  the prior worker's dispatch goroutine has exited by the time we re-enter Start (`startMu`
  ensures sequencing, the dead worker's stdout is closed).

- **`Call` ignores `result` if it's nil.** A caller that doesn't need the return value (e.g.
  a Python function that mutates global state) passes `nil` and avoids the json.Unmarshal.

- **Python prints + module side effects go to the ring buffer, not the Go logger.** Print
  statements in a user's Python function land in `stdout-a`/`stdout-b`. Operational logging
  on the Python side should write to stderr (also captured in the ring), or return data in
  the call result and let Go log it.

- **`Start` does not pre-warm the worker for tests.** Each test that spins up a `Venv` pays
  the full bootstrap cost (`python3 -m venv` is ~1s on a fast Mac, slower on Linux). Tests
  that don't need a real venv use the unit-test entry points (`TestNew_*`, `TestCall_NotReady`,
  etc.) which never call `Start`.

- **`hasPython3` skips, doesn't fail.** Tests that need a real Python skip when `python3 -m
  venv --help` fails, so the suite stays green on minimal CI images without `python3-venv`.
  An accidental skip in a real CI is annoying — verify locally with `go test ./...` showing
  the integration tests are PASS not SKIP.

- **`Logger` uses `LogInfo` / `LogError` rather than slog-style `Info` / `Error`.** The
  `Log`-prefixed names match a common framework convention (so a structured-logger value from
  such a framework can be passed directly without an adapter). Consumers using slog directly
  wrap it in a small adapter; the surface is two methods.
