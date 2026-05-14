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
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

// Frame types crossing the bridge. Mirrored in worker.py.
const (
	frameDefine   = "define"
	frameCall     = "call"
	frameReady    = "ready"
	frameOpDone   = "op_done"
	frameCallDone = "call_done"
	frameError    = "error"
)

// writeFrame encodes obj as one length-prefixed JSON frame on w. The caller is responsible for
// serializing concurrent writes so the (4-byte header, body) pair stays atomic on the wire.
func writeFrame(w io.Writer, obj map[string]any) error {
	body, err := json.Marshal(obj)
	if err != nil {
		return fmt.Errorf("marshal frame: %w", err)
	}
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(body)))
	if _, err := w.Write(header[:]); err != nil {
		return fmt.Errorf("write frame header: %w", err)
	}
	if _, err := w.Write(body); err != nil {
		return fmt.Errorf("write frame body: %w", err)
	}
	return nil
}

// readFrames pulls length-prefixed JSON frames from r and hands each to onFrame until r reports
// EOF or another error. Returns the terminating error (nil on clean EOF).
func readFrames(r io.Reader, onFrame func(map[string]any)) error {
	br := bufio.NewReader(r)
	for {
		var header [4]byte
		if _, err := io.ReadFull(br, header[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil
			}
			return err
		}
		n := binary.BigEndian.Uint32(header[:])
		body := make([]byte, n)
		if _, err := io.ReadFull(br, body); err != nil {
			return err
		}
		var frame map[string]any
		if err := json.Unmarshal(body, &frame); err != nil {
			// Skip malformed frames rather than tearing down the worker.
			continue
		}
		onFrame(frame)
	}
}
