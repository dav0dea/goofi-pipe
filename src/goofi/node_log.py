"""Per-node stdout/stderr capture + a peer-to-peer SSE log server.

The browser's Console shows every node's stdout/stderr. Rather than relay log
bytes through the manager (an unwanted middleman) or push them over iceoryx2
(a fixed-size zero-copy bus built for the data plane, not text streams), each
*process* that hosts node(s) runs one tiny HTTP server that exposes an
**SSE** (Server-Sent Events) endpoint per node: ``GET /<node_id>``. The frontend
connects directly with a native ``EventSource``; the manager only advertises the
endpoint address over the control plane (see ``Node._push_state``).

Capture is process-global but attributes each write to a node:

- a thread-local ``node_id`` is bound at the top of every node-owned thread
  (processing / setup / messaging loops). The redirector reads it to decide
  which node a ``print`` belongs to.
- a *process default* node id is set only in single-node multiprocessing
  children (``_run_node_process``), so threads a node spawns itself still
  attribute. It is deliberately left unset in group-host and LOCAL processes,
  where attribution must be purely thread-local.
- writes from unmapped threads (the manager's own threads, the group-host
  dispatcher, the SSE server threads) fall through to the real stream and are
  **not** captured.

The redirector always **writes through** to the original stream, so the
launching terminal still shows node output. It is only installed when capture
is enabled (non-headless); under ``--headless`` prints stay entirely native.

Only Python-level ``sys.stdout``/``sys.stderr`` writes are captured; output a C
extension makes straight to fd 1/2 is not (it still reaches the terminal).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque, Dict, List, Optional
from urllib.parse import quote, unquote

# Per-node ring-buffer depth — bounds memory and is what a freshly-connected
# Console replays as backfill. The authoritative, much larger history lives in
# the browser (its 100k cap); this is just recent context.
MAX_RECORDS = 5000

# Cap on an un-terminated (no newline) accumulated line so a `\r`-style progress
# stream can't grow without bound; flushed as a line once exceeded.
MAX_PARTIAL = 8192

# SSE keepalive cadence — also how we notice a client disconnected while idle
# (the write to a dead socket raises and ends the stream).
KEEPALIVE_S = 15.0


# ---------------------------------------------------------------------------
# Module state (process-global, lock-guarded)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_enabled = False
_installed = False
_orig_stdout = None
_orig_stderr = None
_process_default: Optional[str] = None
_buffers: Dict[str, "_NodeBuffer"] = {}
_server: Optional["_LogServer"] = None
_server_port: int = 0

# Per-thread attribution + a re-entrancy guard so the emit path can't recurse
# back into the redirector.
_tls = threading.local()


class _NodeBuffer:
    """A node's recent log lines + the synchronisation an SSE handler waits on.

    One per node. ``feed`` is called from the node's threads (under the
    redirector's ``_in_emit`` guard); ``records``/``cond`` are read by the SSE
    handler threads. Lines are assigned a per-node monotonic ``seq`` so the
    frontend can dedupe a replay against what it already holds.
    """

    __slots__ = ("node_id", "cond", "records", "closed", "_seq", "_partial")

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.cond = threading.Condition()
        self.records: Deque[dict] = deque(maxlen=MAX_RECORDS)
        self.closed = False
        self._seq = 0
        self._partial: Dict[str, str] = {"stdout": "", "stderr": ""}

    def feed(self, stream: str, text: str) -> None:
        with self.cond:
            acc = self._partial.get(stream, "") + text
            parts = acc.split("\n")
            # Everything before the last element is a completed line; the tail
            # is the unterminated remainder we hold until the next write.
            for line in parts[:-1]:
                self._append(stream, line)
            remainder = parts[-1]
            if len(remainder) > MAX_PARTIAL:
                self._append(stream, remainder)
                remainder = ""
            self._partial[stream] = remainder
            self.cond.notify_all()

    def _append(self, stream: str, text: str) -> None:
        # cond is held by the caller.
        self.records.append(
            {"node": self.node_id, "stream": stream, "text": text, "seq": self._seq, "ts": time.time()}
        )
        self._seq += 1

    def close(self) -> None:
        with self.cond:
            for stream, partial in self._partial.items():
                if partial:
                    self._append(stream, partial)
                    self._partial[stream] = ""
            self.closed = True
            self.cond.notify_all()


class _Tee:
    """A stdout/stderr stand-in: writes through to the original, then captures.

    Wraps one original stream and knows whether it is ``stdout`` or ``stderr``.
    Everything except ``write``/``writelines`` is delegated to the original.
    """

    def __init__(self, orig, stream_name: str) -> None:
        self._orig = orig
        self._stream_name = stream_name

    def write(self, s: str):
        # Write-through first so the terminal is never starved, regardless of
        # capture state or errors below.
        n = self._orig.write(s)
        if not _enabled or getattr(_tls, "in_emit", False):
            return n
        node_id = getattr(_tls, "node_id", None) or _process_default
        if node_id is None:
            return n
        buf = _buffers.get(node_id)
        if buf is None:
            return n
        _tls.in_emit = True
        try:
            buf.feed(self._stream_name, s)
        except Exception:
            pass
        finally:
            _tls.in_emit = False
        return n

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        # Delegate flush, isatty, fileno, encoding, buffer, … to the real stream.
        return getattr(self._orig, name)


class _LogRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence per-request stderr logging
        pass

    def do_GET(self) -> None:
        try:
            node_id = unquote(self.path.lstrip("/"))
            buf = _buffers.get(node_id)
            if buf is None:
                self.send_response(404)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            # Defeat any proxy buffering of the event stream.
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self._stream(buf)
        except (BrokenPipeError, ConnectionResetError, ValueError):
            return
        except Exception:
            return

    def _stream(self, buf: "_NodeBuffer") -> None:
        # Honor SSE resume: on auto-reconnect the browser sends the last id it
        # saw, so we replay only what it missed (no dupes, no gaps).
        last_id = self.headers.get("Last-Event-ID")
        cursor = int(last_id) if last_id and last_id.lstrip("-").isdigit() else -1
        # Replay current backfill past the cursor.
        with buf.cond:
            snapshot = [r for r in buf.records if r["seq"] > cursor]
        for rec in snapshot:
            self._send(rec)
            cursor = rec["seq"]
        # Live tail.
        while True:
            with buf.cond:
                new = [r for r in buf.records if r["seq"] > cursor]
                if not new and not buf.closed:
                    buf.cond.wait(timeout=KEEPALIVE_S)
                    new = [r for r in buf.records if r["seq"] > cursor]
                closed = buf.closed
            if new:
                for rec in new:
                    self._send(rec)
                    cursor = rec["seq"]
            else:
                # Keepalive comment — also surfaces a client disconnect.
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
            if closed and not new:
                return

    def _send(self, rec: dict) -> None:
        payload = json.dumps(rec, ensure_ascii=False)
        self.wfile.write(f"id: {rec['seq']}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()


class _LogServer:
    """One threaded SSE server per process, serving every registered node."""

    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _LogRequestHandler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="goofi-log-sse", daemon=True
        )
        self._thread.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_capture() -> None:
    """Swap ``sys.stdout``/``sys.stderr`` for capturing tees (idempotent)."""
    global _installed, _orig_stdout, _orig_stderr
    with _lock:
        if _installed:
            return
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        sys.stdout = _Tee(_orig_stdout, "stdout")
        sys.stderr = _Tee(_orig_stderr, "stderr")
        _installed = True


def set_process_default_node(node_id: str) -> None:
    """Attribute otherwise-unmapped threads to ``node_id`` (single-node MP)."""
    global _process_default
    _process_default = node_id


def bind_thread_node(node_id: str) -> None:
    """Attribute the *calling* thread's output to ``node_id``."""
    _tls.node_id = node_id


def register_node(node_id: str) -> str:
    """Enable capture, ensure the SSE server, and return this node's endpoint.

    Called by a node only when capture is wanted (non-headless). Returns the
    ``http://127.0.0.1:<port>/<node_id>`` URL the manager advertises.
    """
    global _enabled
    _enabled = True
    install_capture()
    with _lock:
        if node_id not in _buffers:
            _buffers[node_id] = _NodeBuffer(node_id)
        _ensure_server_locked()
        port = _server_port
    return f"http://127.0.0.1:{port}/{quote(node_id, safe='')}"


def unregister_node(node_id: str) -> None:
    """Flush the node's pending line and drop its buffer (node terminated)."""
    with _lock:
        buf = _buffers.pop(node_id, None)
    if buf is not None:
        buf.close()


def _ensure_server_locked() -> None:
    # Caller holds _lock.
    global _server, _server_port
    if _server is None:
        _server = _LogServer()
        _server_port = _server.port


def _reset_for_tests() -> None:
    """Tear capture down — test-only, so a suite doesn't leak global state."""
    global _enabled, _installed, _process_default, _server, _server_port
    with _lock:
        if _installed:
            if _orig_stdout is not None:
                sys.stdout = _orig_stdout
            if _orig_stderr is not None:
                sys.stderr = _orig_stderr
        for buf in list(_buffers.values()):
            buf.close()
        _buffers.clear()
        _enabled = False
        _installed = False
        _process_default = None
        if _server is not None:
            try:
                _server._httpd.shutdown()
                _server._httpd.server_close()
            except Exception:
                pass
        _server = None
        _server_port = 0
