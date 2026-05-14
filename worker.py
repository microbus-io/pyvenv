# Copyright (c) 2026 Microbus LLC and various contributors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# 	http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyvenv worker. Reads length-prefixed JSON frames on stdin, dispatches
# `define` and `call` frames to a ThreadPoolExecutor, writes response
# frames on stdout under a lock.

import argparse
import json
import struct
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

GLOBALS = {}
_stdout_lock = threading.Lock()
_stdout = sys.stdout.buffer
_stdin = sys.stdin.buffer


def write_frame(obj):
    data = json.dumps(obj, default=str).encode("utf-8")
    with _stdout_lock:
        _stdout.write(struct.pack(">I", len(data)))
        _stdout.write(data)
        _stdout.flush()


def read_frame():
    header = _stdin.read(4)
    if not header or len(header) < 4:
        return None
    (n,) = struct.unpack(">I", header)
    body = b""
    while len(body) < n:
        chunk = _stdin.read(n - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body)


def _err_payload(e):
    return {
        "errorType": type(e).__name__,
        "errorMessage": str(e),
        "traceback": traceback.format_exc(),
    }


def handle_define(op_id, code):
    try:
        compiled = compile(code, "<define>", "exec")
        exec(compiled, GLOBALS)
        write_frame({"type": "op_done", "opID": op_id, "ok": True})
    except Exception as e:
        frame = {"type": "op_done", "opID": op_id, "ok": False}
        frame.update(_err_payload(e))
        write_frame(frame)


def _run_call(func_name, args):
    func = GLOBALS.get(func_name)
    if func is None:
        raise NameError("function %r not defined" % func_name)
    return func(args)


def handle_call(executor, call_id, func_name, args):
    def _done(future):
        try:
            result = future.result()
            write_frame({
                "type": "call_done",
                "callID": call_id,
                "ok": True,
                "result": result,
            })
        except Exception as e:
            frame = {"type": "call_done", "callID": call_id, "ok": False}
            frame.update(_err_payload(e))
            write_frame(frame)

    future = executor.submit(_run_call, func_name, args)
    future.add_done_callback(_done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=1)
    opts = parser.parse_args()

    executor = ThreadPoolExecutor(max_workers=opts.max_workers)

    write_frame({"type": "ready"})

    try:
        while True:
            frame = read_frame()
            if frame is None:
                break
            t = frame.get("type")
            if t == "define":
                handle_define(frame.get("opID"), frame.get("code", ""))
            elif t == "call":
                handle_call(executor, frame.get("callID"), frame.get("func"), frame.get("args"))
            elif t == "ping":
                write_frame({"type": "pong"})
            else:
                write_frame({
                    "type": "error",
                    "errorType": "ProtocolError",
                    "errorMessage": "unknown frame type %r" % t,
                })
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
