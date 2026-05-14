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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// ringWriter is a double-buffered on-disk writer. Two files alternate as the active buffer; once
// the active file fills its cap, writes switch to the other. The previous file is preserved until
// the new one fills, then overwritten. At any moment readers can access between cap and 2*cap
// bytes of history.
type ringWriter struct {
	dir      string
	base     string
	capBytes int

	mu      sync.Mutex
	active  *os.File
	other   string
	written int
	activeP string
}

func newRingWriter(dir, base string, capBytes int) (*ringWriter, error) {
	if capBytes <= 0 {
		capBytes = 256 * 1024
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create ring dir %s: %w", dir, err)
	}
	path := filepath.Join(dir, base+"-a")
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create ring file %s: %w", path, err)
	}
	return &ringWriter{
		dir:      dir,
		base:     base,
		capBytes: capBytes,
		active:   f,
		activeP:  path,
	}, nil
}

// Write appends to the active buffer, rotating as needed. If a single write exceeds capBytes,
// only the last capBytes bytes are retained.
func (r *ringWriter) Write(p []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active == nil {
		return 0, errors.New("ring writer closed")
	}
	if len(p) > r.capBytes {
		p = p[len(p)-r.capBytes:]
	}
	if r.written+len(p) > r.capBytes {
		if err = r.rotateLocked(); err != nil {
			return 0, err
		}
	}
	n, err = r.active.Write(p)
	r.written += n
	return n, err
}

func (r *ringWriter) rotateLocked() error {
	if err := r.active.Close(); err != nil {
		return err
	}
	var nextName string
	if filepath.Base(r.activeP) == r.base+"-a" {
		nextName = r.base + "-b"
	} else {
		nextName = r.base + "-a"
	}
	r.other = r.activeP
	r.activeP = filepath.Join(r.dir, nextName)
	f, err := os.Create(r.activeP)
	if err != nil {
		return err
	}
	r.active = f
	r.written = 0
	return nil
}

// Tail returns the older buffer concatenated with the active buffer. Returns up to 2*capBytes
// bytes (older first, then active).
func (r *ringWriter) Tail() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	var older, active []byte
	if r.other != "" {
		if data, err := os.ReadFile(r.other); err == nil {
			older = data
		}
	}
	if r.active != nil {
		if err := r.active.Sync(); err == nil {
			if data, err := os.ReadFile(r.activeP); err == nil {
				active = data
			}
		}
	}
	out := make([]byte, 0, len(older)+len(active))
	out = append(out, older...)
	out = append(out, active...)
	return out
}

// Close releases the active file handle. The on-disk files remain.
func (r *ringWriter) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil {
		err := r.active.Close()
		r.active = nil
		return err
	}
	return nil
}
