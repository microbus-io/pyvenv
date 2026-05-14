# pyvenv

Run a Python virtual environment as an in-process worker for a Go program.

`pyvenv` spawns one long-lived Python subprocess per `Venv`, talks to it over
length-prefixed JSON frames on stdin/stdout, and exposes typed `Call`/`Await`
into named Python functions. The Go side stays in charge of all concerns
(logging, tracing, retries, downstream calls); Python is a pure compute
kernel. `Call` returns immediately with a `callID`; `Await(ctx, callID)`
blocks until the call completes. A ctx-expired `Await` does not consume the
in-flight call — a later `Await` on the same `callID` re-enters the wait, so
Python work can outlive any single caller's ctx (e.g. a workflow task's
per-step time budget).

The package is framework-agnostic: its only dependency is the Go standard
library. The intended consumer is any Go program that wants to call into
Python libraries (PyTorch, pandas, scikit-learn, sentence-transformers, etc.)
without reimplementing subprocess management.

## Install

```sh
go get github.com/microbus-io/pyvenv
```

Requires `python3` (and the standard library `venv` module) on the host's
PATH at runtime, or an explicit `PythonInterpreter` config value.

## Usage

```go
package main

import (
    "context"
    "embed"
    "fmt"

    "github.com/microbus-io/pyvenv"
)

//go:embed *.py
var pyFiles embed.FS

func main() {
    ctx := context.Background()

    helpers, _ := pyFiles.ReadFile("helpers.py")
    service, _ := pyFiles.ReadFile("service.py")

    v, err := pyvenv.New(pyvenv.Config{
        Sources:      []string{string(helpers), string(service)},
        Requirements: []string{"sentence-transformers"},
        MaxWorkers:   2,
    })
    if err != nil {
        panic(err)
    }
    defer v.Close(ctx)

    if err := v.Start(ctx); err != nil {
        panic(err)
    }

    var out struct {
        Vector []float64 `json:"vector"`
    }
    // Synchronous convenience: Call + Await on the same goroutine.
    err = v.CallAndAwait(ctx, "embed", map[string]any{"text": "hello world"}, &out)
    if err != nil {
        panic(err)
    }
    fmt.Println(out.Vector)

    // Or split for callers that need to outlive their ctx (e.g. workflow tasks):
    callID, _ := v.Call(ctx, "embed", map[string]any{"text": "another"})
    // ... store callID somewhere durable, return early, come back later ...
    _ = v.Await(ctx, callID, &out)
}
```

A matching `service.py`:

```python
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed(args):
    vec = MODEL.encode(args["text"]).tolist()
    return {"vector": vec}
```

## Lifecycle

| Phase | What happens |
|---|---|
| `New` | Validate config; no subprocess yet. |
| `Start` | Create venv on disk (if absent) via `python3 -m venv`. Run `pip install` if `Requirements` changed since last successful install. Spawn `worker.py` under the venv's Python. Load `Sources` via one `define` frame. Block until the worker emits its initial `ready` frame. |
| `Call` | Marshal `args` to JSON, send a `call` frame, return a `callID`. Does not block on Python execution. |
| `Await` | Block until the call identified by `callID` completes or ctx expires. On success, unmarshal the result and consume the entry — subsequent Awaits on the same callID return `ErrUnknown`. On ctx expiry, do not consume. |
| `Result` | Non-blocking peek. Returns `(false, nil)` while the call is in flight, `(true, nil)` on success with the result populated, `(true, <python error>)` on Python failure, `(false, ErrUnknown)` when the callID is unknown. |
| `CallAndAwait` | Convenience: `Call` followed by `Await` on the same goroutine. Synchronous shape. |
| `Close` | Kill the subprocess, wake parked Awaits with `ErrClosed`, remove `BaseDir` on disk. Idempotent. |

`Start` is safe to call again after `StateDied`. The on-disk venv is reused, `pip install` is skipped if requirements are unchanged, and a fresh worker is spawned.

## Liveness

```go
v, _ := pyvenv.New(pyvenv.Config{
    // ...
    LivenessCallback: func(state pyvenv.State, err error) {
        switch state {
        case pyvenv.StateReady:
            // Worker is up. Activate dependent features.
        case pyvenv.StateDied:
            // Subprocess crashed. Recover (e.g. v.Start(ctx) again).
        }
    },
})
```

`LivenessCallback` fires only on async transitions:

- `StateReady` after a successful `Start`.
- `StateDied` when the subprocess exits unexpectedly (not when `Close` killed it).

`Start` failures surface as the synchronous error from `Start` — no callback for the never-ready case.

## Wire protocol

`pyvenv` and `worker.py` exchange length-prefixed JSON frames. A 4-byte
big-endian unsigned length prefixes each JSON body.

Go → Python:

```json
{"type": "define", "opID": "<id>", "code": "<python source>"}
{"type": "call",   "callID": "<id>", "func": "<name>", "args": <any JSON>}
{"type": "ping"}
```

Python → Go:

```json
{"type": "ready"}
{"type": "op_done",   "opID":   "<id>", "ok": true}
{"type": "op_done",   "opID":   "<id>", "ok": false, "errorType": "...", "errorMessage": "..."}
{"type": "call_done", "callID": "<id>", "ok": true,  "result": <any JSON>}
{"type": "call_done", "callID": "<id>", "ok": false, "errorType": "...", "errorMessage": "..."}
{"type": "pong"}
```

Frames are exchanged over `worker.py`'s stdin/stdout. Stderr is captured into a per-stream ring buffer accessible via `TailStdErr`.

## Result cache

A completed call's result is held in an in-memory cache keyed by `callID`
until either a successful `Await` / `Result` consumes it, or `Config.ResultCacheTTL`
elapses (default 15 minutes; set negative to disable eviction). A consumed
result frees the entry immediately; the TTL is the safety net for callers
that never come back to claim their result.

Single-delivery semantics: only one successful `Await` (or `Result`) per
`callID`. Concurrent waiters would have one win the delivery and the other
receive `ErrUnknown`. The framework-side workflow primitives don't normally
create concurrent Awaits — retries within a flow are sequential and Fork
creates a fresh task that issues its own `Call` — so this is a documented
library-level constraint rather than a workflow-author concern.

## Concurrency

One Python process holds one GIL, but the GIL is released during native
calls (PyTorch, NumPy, pandas) and during I/O. Concurrent `Call`s into the
same `Venv` run on threads in a `ThreadPoolExecutor` sized by
`Config.MaxWorkers`.

Two `Call`s with the same function name and different args run on separate
worker threads in parallel, assuming the Python code is thread-safe. The
caller is responsible for ensuring that.

For pure-Python CPU-bound work that won't release the GIL, the right answer
is one `Venv` per worker, not multiple workers within one `Venv`. Run
several `Venv` instances in your Go program if you need process-level
parallelism.

## Output buffers

Each `Venv` has rolling stdout and stderr buffers on disk under
`BaseDir/meta/`. `TailStdOut()` and `TailStdErr()` return up to
`2 * OutputBufferKB` bytes of recent output. Pull-based: callers only pay
for output they actually fetch.

Output from concurrent `Call`s interleaves in the buffer. That is expected
— per-call tagging would require per-call buffers, which scales worse.

## License

Apache 2.0.
