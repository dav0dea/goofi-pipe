# goofi-pipe Peer-to-Peer Viewer-Data Plane with Node-Side Thalamus Reduction

## 1. Goal and Topology

### Goal

Move the **viewer-data path** of goofi-pipe to **peer-to-peer**: the browser connects
**directly to the node-host process** that produces a slot's data, and the data is
**reduced inside the node process** (a node-side "thalamus") to exactly what the viewer
can display before it is encoded and sent. The **manager is removed from the data path**
entirely; it only *advertises* per-process data endpoints over the existing `/control`
WebSocket, exactly the way it already advertises per-node SSE **log** endpoints today.
This mirrors `src/goofi/node_log.py`, the shipped, proven template for a per-process
peer-to-peer server.

### Why the data path routes through the manager today (dearpygui legacy)

The current data plane is a leftover of the dearpygui era, when the manager process
*was* the UI. The browser bridge inherited that shape:

1. Browser opens one WS per `(node, slot)` to the **manager** at
   `ws://<host>/data/<node>/<slot>` (`frontend/src/lib/api/data.ts:21-22`).
2. The bridge's `DataHub.handler` (`src/goofi/bridge/data.py:89`) registers
   `ref.set_data_handler(slot, on_frame)` (`data.py:121`).
3. `NodeRef.set_data_handler` (`src/goofi/node_helpers.py:400`) opens the **manager's
   own** iceoryx2 subscriber to the node's output (`open_output_subscriber`,
   `node_helpers.py:392-398`) and starts a per-NodeRef `_data_pump` thread
   (`node_helpers.py:437`) that **decodes every frame in the manager process**
   (`node_helpers.py:462`).
4. `DataHub.on_frame` (`data.py:110`) then **re-encodes the full, unreduced `Data`**
   and pushes it over the browser WS.

So every viewed frame costs: node-encode → iceoryx2 SHM copy → **manager decode** →
**manager re-encode** → WS → browser decode. Three codec passes plus a full cross-process
SHM copy of **unreduced** data (one 44.1 kHz / 60 s mono f32 buffer is **10.6 MB**;
~318 MB/s at 30 Hz). The reduction the viewer actually needs (e.g. → ~2000 points,
**8 KB**, ~1300× shrink) happens *nowhere*.

### Target

- The producing **node process** hosts a tiny binary WebSocket server (one per host
  process), advertises its base URL as `data_endpoint` via `STATE_UPDATE`.
- The browser discovers `data_endpoint` from the control plane (like `log_endpoint`),
  connects **directly** to the node process, and sends a **ViewSpec** describing what it
  can display.
- The node **reduces** the live `Data` to the folded ViewSpec **before** encoding, and
  streams the small reduced frames straight to the browser.
- The manager only relays the endpoint string. The node↔node iceoryx2 path and
  `codec.py` are **untouched**.

---

## 2. Target Architecture (ASCII)

```
                            ┌──────────────────────── browser tab ───────────────────────┐
                            │  SvelteKit SPA                                              │
                            │   ┌─ /control WS ──────────────► manager (graph + events)   │
                            │   │      hello / state_update {data_endpoint, log_endpoint} │
                            │   │                                                          │
                            │   │  dataStream store (mirrors logStream): RESOLVES          │
                            │   │  node.data_endpoint AND OWNS the per-(node,slot) WS      │
                            │   │  thalamus store: folds ViewerFeed consumers → ViewSpec   │
                            │   ▼                                                          │
                            │  per (node,slot) WS  ───────────────────────┐                │
                            └──────────────────────────────────┬──────────┼────────────────┘
                                                               │          │
   ┌──────────── manager process ────────────┐                │ (control)│ (data, P2P)
   │  Manager  (graph, links, persistence)    │◄───────────────┘          │
   │  bridge/server.py  (HTTP + /control WS)   │                           │
   │  bridge/control.py (RPC + state relay)    │   advertises only the     │
   │  ── NO data path ──                       │   data_endpoint string    │
   │  (DataHub / _data_pump REMOVED)           │                           │
   └───────────────────┬──────────────────────┘                           │
                       │ ctrl pub / status sub (iceoryx2)                  │
                       ▼                                                   ▼
   ┌──────────── node-host process (1..N nodes) ──────────────────────────────────────┐
   │  node_data.py: ONE ThreadingHTTPServer @ 127.0.0.1:0 (per process)                │
   │     ws://127.0.0.1:<port>/<node_id>/<slot>  (hand-rolled RFC6455; binary DOWN,     │
   │                                              text ViewSpec UP; CORS *)             │
   │     PER-CONNECTION latest-wins byte mailbox; registry: (node,slot)->[mailboxes]    │
   │     PER-(node,slot) folded ViewSpec (last-received-wins)                           │
   │                                                                                    │
   │  Node._processing_loop (node.py:730):                                              │
   │     if subscriber_count==0 and viewer_count==0: continue                           │
   │     data = Data(slot.dtype, value[0], value[1])   # node.py:737  (live, pre-encode)│
   │     ├─ if subscriber_count>0:  prepare_encode(data); iceoryx2 publish FULL  ◄── UNCHANGED
   │     └─ if viewer_count>0:      reduced = reduce_for_view(data, spec)               │
   │                                buf = encode_data(reduced)  # eager bytes           │
   │                                node_data.publish(node,slot,buf)  # fan to mailboxes │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sequence Diagrams (ASCII)

### (a) Browser opens a viewer → discover → connect → negotiate → receive frames

```
ViewerFeed     thalamus store      dataStream(owns WS)    graph store    manager(/control)   node-host data server   node _processing_loop
   │ visible+expand (IO/RO)│              │                  │                │                      │                       │
   ├─ addConsumer(spec) ──►│              │                  │                │                      │                       │
   │                       ├─ viewSpecFor(node,slot) (fold, debounce 150ms)   │                      │                       │
   │                       │              │◄── read graph().node.data_endpoint (null until 1st STATE_UPDATE)              │
   │                       │              │                  │◄── state_update {data_endpoint} ─────┤ (advertised)         │
   │                       │              │                  ├─ merge data_endpoint                 │                       │
   │                       ├─ note need ─►│ open WS <base>/<node>/<slot>?spec=<b64url(ViewSpec)> ───────────────────────────►│ upgrade 101 (CORS *)
   │                       │              │                  │                │                      ├─ viewer_count += 1 (lock)
   │                       │              │                  │                │                      ├─ +wake_processing (0→1)
   │                       │              │                  │                │                      ├─ _specs[(node,slot)] = seed
   │                       │              │ send TEXT {op:"view",spec,v:1} ─────────────────────────────────────────────────►│ _specs[(node,slot)] = spec (LWW)
   │                       │              │                  │                │                      │                       ├─ gate: subscriber||viewer>0 ⇒ produce
   │                       │              │                  │                │                      │                       ├─ reduce_for_view(data, spec)
   │                       │              │                  │                │                      │                       ├─ encode_data(reduced)
   │                       │              │                  │                │                      │◄ publish(node,slot,buf)┤ fan to each conn mailbox
   │◄── onFrame(decoded) ◄ RAF coalesce ◄ binary WS frame ◄──────────────────────────────────────────────────────────────────┤ handler writes latest
```

### (b) Collapse / disconnect

```
ViewerFeed        thalamus/dataStream         node-host data server        node _processing_loop
   │ hidden OR collapsed │                            │                            │
   ├─ removeConsumer ───►│                            │                            │
   │                     ├─ no more needs? close WS ─►│ socket.shutdown(SHUT_RDWR) │
   │                     │                            ├─ reader thread unblocks+exits
   │                     │                            ├─ handler loop exits         │
   │                     │                            ├─ finally: viewer_count -=1 (lock, max(0,…))
   │                     │                            │   (also on BrokenPipe/Reset)│
   │                     │                            ├─ drop this conn's mailbox    │
   │                     │                            │                            ├─ gate false ⇒ slot skipped
   │                     │                            │                            │   (unless a node consumer remains)
```

### (c) Node restart / reconnect (new ephemeral port)

```
graph store          dataStream(owns WS)          old node-host (dead)     new node-host
   │ node removed/re-added │                            │                       │
   │ OR new STATE_UPDATE   │                            │                       │
   │  data_endpoint CHANGED│                            │                       │
   ├─ merge new endpoint ─►│                            │                       │
   │                       ├─ reconcile: endpoints.get(name) !== ws.endpoint     │
   │                       ├─ close stale WS (was retrying dead port) ──────────►X (gone)
   │                       ├─ open WS to NEW <base>/<node>/<slot>?spec ──────────────────────────►│ 101
   │                       ├─ re-send {op:"view",spec,v:1}  (no server resume) ──────────────────►│ _specs set (LWW)
```

> Reconnect rule: the data WS has **no resume buffer** (latest-wins, history-less). On
> every (re)connect the client **re-resolves `data_endpoint` from graph state** (never
> retries a stale port) and **re-sends its ViewSpec** (seed query + immediate
> `{op:"view"}`).

---

## 4. Data-Server Design (`src/goofi/node_data.py`, NEW)

A structural twin of `src/goofi/node_log.py`, differing in: binary WebSocket transport
(not SSE text), per-`(node, slot)` granularity, an inbound ViewSpec channel, **per-
connection** mailboxes, and `viewer_count` bookkeeping on the `OutputSlot`.

### 4.0 Dependency surface (PINNED)

- `node_data.py`: **stdlib only** (`http.server`, `socket`, `threading`, `hashlib`,
  `base64`, `struct`, `json`, `urllib.parse`). **No** `websockets`, **no** `aiohttp`,
  **no** asyncio loop in node processes (heavy per-process cost; in `--no-multiprocessing`
  / LOCAL mode it would collide with the manager's own aiohttp app).
- `node_reduce.py`: **numpy only**. **No PIL, no scipy, no cv2.** All reductions are
  pure-numpy.

### 4.1 Transport: hand-rolled RFC6455 over stdlib `ThreadingHTTPServer`

- One process-global `ThreadingHTTPServer(("127.0.0.1", 0), _DataRequestHandler)` with
  `daemon_threads = True` (copy `node_log.py:236-243`). `127.0.0.1` only; OS-assigned
  ephemeral port read from `server_address[1]`. Thread name `"goofi-data-ws"` (distinct
  from log server's `"goofi-log-sse"`).
- `do_GET` performs the RFC6455 upgrade **by hand** (see §4.3 + §4.5 for full bodies).
  - `Sec-WebSocket-Accept = base64(sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))`.
  - `101` response with `Upgrade: websocket`, `Connection: Upgrade`,
    `Sec-WebSocket-Accept: <accept>`, **and `Access-Control-Allow-Origin: *`** (CORS is
    mandatory — page is served from the manager origin, connects to the node port).
  - Non-WS GET (no `Upgrade`) → `404` with `Access-Control-Allow-Origin: *`
    (mirror `node_log.py:178-182`).
- **Raw-socket I/O, never the buffered `wfile`/`rfile`.** Use `self.connection` for both
  directions. A buffered reader/writer shared across two threads is not thread-safe.
- **Single write owner via a per-connection write lock.** The handler thread drains the
  mailbox and writes binary frames; the reader thread may need to write a `pong` (reply
  to client `ping`) or a `close` echo. **Every** socket write (binary frame, ping,
  pong, close) goes through `_send_frame(...)` which takes the per-connection
  `_write_lock`. This makes concurrent writes from the two threads safe.
- **Reader-thread teardown contract (prevents thread leak per viewer churn).** When the
  handler exits (mailbox closed, broken pipe, or normal close), it calls
  `self.connection.shutdown(socket.SHUT_RDWR)` in a `finally`. This unblocks the reader
  thread's blocking `recv`, which then sees EOF/`OSError` and terminates. The reader is a
  daemon thread; the handler does not `join` it (avoids shutdown deadlock) but the
  `shutdown` guarantees it stops blocking.

### 4.2 Module state + public API (full sketch)

```python
# src/goofi/node_data.py
from __future__ import annotations
import base64, hashlib, json, socket, struct, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse, parse_qs
from goofi.node_reduce import ViewSpec, viewspec_from_dict

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_SEED_MAX = 4096            # max length of the ?spec= base64url payload

_lock = threading.Lock()
_server: Optional["_DataServer"] = None
_server_port: int = 0

# node_id -> Node instance (so a handler can reach slot.viewer_count / viewer_lock).
_nodes: Dict[str, "Node"] = {}

# (node_id, slot) -> list of live per-connection mailboxes (fan-out).
_mailboxes: Dict[Tuple[str, str], List["_ConnMailbox"]] = {}
_mailboxes_lock = threading.Lock()

# (node_id, slot) -> folded ViewSpec. ONE per slot; folding is frontend-side.
# Last-received-wins (no version comparison server-side). Atomic dict ref-write under GIL.
_specs: Dict[Tuple[str, str], ViewSpec] = {}


class _ConnMailbox:
    """One latest-wins binary frame for ONE connection + Condition."""
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self._pending: Optional[bytes] = None
        self.closed = False

    def push(self, frame: bytes) -> None:
        with self.cond:
            self._pending = frame            # overwrite ⇒ drop-oldest / latest-wins
            self.cond.notify_all()

    def take(self, timeout: float) -> Optional[bytes]:
        with self.cond:
            if self._pending is None and not self.closed:
                self.cond.wait(timeout)
            frame, self._pending = self._pending, None
            return frame

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


class _DataServer:
    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DataRequestHandler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever,
                         name="goofi-data-ws", daemon=True).start()


def register_node(node_id: str, node: "Node") -> str:
    """Ensure the per-process server, register the node, return its base URL.

    Mirrors node_log.register_node. ONE call site: Node.__init__ (see §7.1).
    All three host contexts (single-node MP, group host, LOCAL) construct the
    Node, so they all hit this one line — do NOT edit _run_node_process /
    _spawn_local / create_local.
    """
    with _lock:
        _nodes[node_id] = node
        _ensure_server_locked()
        port = _server_port
    return f"ws://127.0.0.1:{port}/{node_id}"


def unregister_node(node_id: str) -> None:
    """Drop the node; close+wake all its connection mailboxes (node terminated)."""
    with _lock:
        _nodes.pop(node_id, None)
    with _mailboxes_lock:
        dead = [k for k in _mailboxes if k[0] == node_id]
        for k in dead:
            for mb in _mailboxes.pop(k):
                mb.close()
    for k in [k for k in _specs if k[0] == node_id]:
        _specs.pop(k, None)


def viewspec_for(node_id: str, slot: str) -> Optional[ViewSpec]:
    return _specs.get((node_id, slot))           # atomic dict read under GIL


def publish(node_id: str, slot: str, frame: bytes) -> None:
    """Called from the node's _processing_loop thread. Fan immutable bytes to
    every live connection mailbox for this slot (pointer-swap + notify)."""
    with _mailboxes_lock:
        boxes = list(_mailboxes.get((node_id, slot), ()))
    for mb in boxes:
        mb.push(frame)


def _ensure_server_locked() -> None:
    global _server, _server_port
    if _server is None:
        _server = _DataServer()
        _server_port = _server.port


def _reset_for_tests() -> None:
    """Test-only: shut the server, free the port, clear globals (mirror node_log)."""
    global _server, _server_port
    with _mailboxes_lock:
        for boxes in _mailboxes.values():
            for mb in boxes:
                mb.close()
        _mailboxes.clear()
    _specs.clear()
    with _lock:
        _nodes.clear()
        if _server is not None:
            try:
                _server._httpd.shutdown()
                _server._httpd.server_close()
            except Exception:
                pass
        _server = None
        _server_port = 0
```

> **Why per-connection mailboxes (not per-slot).** Two browser tabs (or an overlapping
> reconnect) on one `(node, slot)` would, with a single shared mailbox, **steal frames
> from each other** (`take` pops `_pending` to `None`). Per-connection mailboxes + a
> `publish` fan-out give every connection every latest frame. The frontend still dedups
> to one WS per `(node, slot)` *per tab*, so the common case is one mailbox.

### 4.3 Request handler (full sketch)

```python
class _DataRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = unquote(parsed.path.lstrip("/")).split("/", 1)
        if len(parts) != 2:
            return self._reject(404)
        node_id, slot = parts[0], parts[1]
        node = _nodes.get(node_id)
        if node is None or slot not in node.output_slots:
            return self._reject(404)
        if self.headers.get("Upgrade", "").lower() != "websocket":
            return self._reject(404)
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            return self._reject(400)

        accept = base64.b64encode(
            hashlib.sha1((key + _GUID).encode()).digest()).decode()
        self._raw_send(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept.encode() + b"\r\n"
            b"Access-Control-Allow-Origin: *\r\n\r\n")

        out_slot = node.output_slots[slot]
        skey = (node_id, slot)
        self._write_lock = threading.Lock()

        # First-frame seed: ?spec=<base64url(json)>. Last-received-wins.
        seed = _parse_seed(parse_qs(parsed.query).get("spec", [None])[0])
        if seed is not None:
            _specs[skey] = seed

        mb = _ConnMailbox()
        with _mailboxes_lock:
            _mailboxes.setdefault(skey, []).append(mb)

        first = False
        with out_slot.viewer_lock:
            out_slot.viewer_count += 1
            first = out_slot.viewer_count == 1
        if first:
            node._wake_processing()      # node.py:312 — start an idle leaf producer

        reader = threading.Thread(target=self._read_inbound, args=(skey,), daemon=True)
        reader.start()
        try:
            while True:
                frame = mb.take(timeout=15.0)
                if mb.closed:
                    break
                if frame is None:
                    self._send_frame(0x9, b"")     # ping keepalive + dead-socket detect
                    continue
                self._send_frame(0x2, frame)        # binary, 64-bit length path for >64KB
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with out_slot.viewer_lock:
                out_slot.viewer_count = max(0, out_slot.viewer_count - 1)
            with _mailboxes_lock:
                lst = _mailboxes.get(skey)
                if lst and mb in lst:
                    lst.remove(mb)
                    if not lst:
                        _mailboxes.pop(skey, None)
            try:
                self.connection.shutdown(socket.SHUT_RDWR)   # unblock reader
            except OSError:
                pass

    def _read_inbound(self, skey) -> None:
        try:
            for op, payload in self._ws_frames():   # unmasks client frames
                if op == 0x8:                        # close
                    self._send_frame(0x8, b"")
                    break
                if op == 0x9:                        # ping → pong
                    self._send_frame(0xA, payload)
                elif op == 0x1:                      # text = ViewSpec
                    spec = _parse_view_message(payload)
                    if spec is not None:
                        _specs[skey] = spec          # last-received-wins, atomic ref swap
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            pass

    # ---- helpers --------------------------------------------------------
    def _reject(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _raw_send(self, b: bytes) -> None:
        self.connection.sendall(b)
```

### 4.4 ViewSpec wire helpers (full bodies)

```python
def _parse_seed(b64: Optional[str]) -> Optional[ViewSpec]:
    if not b64 or len(b64) > _SEED_MAX:
        return None
    try:
        pad = "=" * (-len(b64) % 4)
        raw = base64.urlsafe_b64decode(b64 + pad)
        return viewspec_from_dict(json.loads(raw.decode("utf-8")))
    except Exception:
        return None

def _parse_view_message(payload: bytes) -> Optional[ViewSpec]:
    try:
        msg = json.loads(payload.decode("utf-8"))
        if isinstance(msg, dict) and msg.get("op") == "view":
            return viewspec_from_dict(msg.get("spec") or {})
    except Exception:
        return None
    return None
```

> **No server-side version comparison.** The server stores the **last-received** ViewSpec
> per slot (last-writer-wins). The `v` field is carried for the **client's** ordering only
> and is ignored by the node. This is sound because (a) the frontend folds all consumers
> into ONE spec and debounces it (150 ms, §8.2), so the server sees a low-rate, already-
> coalesced spec stream, and (b) latest-wins on the next frame self-corrects any transient.
> This deliberately drops the contradictory "monotone per connection compared against a
> per-slot store" rule from earlier drafts.

### 4.5 RFC6455 framing (full bodies — isolate + unit-test at 100 B / 70 KB / 3 MB)

```python
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        # Server→client frames are NOT masked. 64-bit length is MANDATORY for
        # reduced HD frames (~2.7 MB > 65535). FIN=1 always (no fragmentation out).
        n = len(payload)
        header = bytearray()
        header.append(0x80 | (opcode & 0x0F))
        if n <= 125:
            header.append(n)
        elif n <= 0xFFFF:
            header.append(126)
            header += struct.pack("!H", n)
        else:
            header.append(127)
            header += struct.pack("!Q", n)
        with self._write_lock:
            self.connection.sendall(bytes(header) + payload)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.connection.recv(n - len(buf))
            if not chunk:
                raise ConnectionResetError
            buf += chunk
        return bytes(buf)

    def _ws_frames(self):
        """Yield (opcode, unmasked_payload) for each inbound client frame.
        Client→server frames are ALWAYS masked (RFC6455 §5.3); we unmask."""
        MAX_IN = 1 << 20   # 1 MB cap on inbound (ViewSpec is tiny); guard hang/DoS
        while True:
            b0, b1 = self._recv_exact(2)
            opcode = b0 & 0x0F
            masked = (b1 & 0x80) != 0
            ln = b1 & 0x7F
            if ln == 126:
                ln = struct.unpack("!H", self._recv_exact(2))[0]
            elif ln == 127:
                ln = struct.unpack("!Q", self._recv_exact(8))[0]
            if ln > MAX_IN:
                raise ValueError("inbound frame too large")
            mask = self._recv_exact(4) if masked else b"\x00\x00\x00\x00"
            data = bytearray(self._recv_exact(ln))
            if masked:
                for i in range(ln):
                    data[i] ^= mask[i & 3]
            yield opcode, bytes(data)
```

### 4.6 Process-group serving

A `NodeProcess` host runs **many** nodes in one process (`node_helpers.py:544-566`). The
server is **process-global, keyed by `(node_id, slot)`**; the URL embeds `node_id`, so one
port serves all co-hosted nodes. Do **not** create one server per node. Unlike
`node_log`, there is **no thread-attribution dance** (`set_process_default_node` /
`bind_thread_node`): the producing node is identified explicitly by the URL path and the
`publish(node_id, slot, …)` call.

### 4.7 Three host contexts that bootstrap the server (ONE call site)

Registration is **one line in `Node.__init__`** (§7.1), under the existing `capture_logs`
gate, beside `node_log.register_node` (`node.py:173-174`). All three contexts construct the
`Node`, so all three get a server **without editing the entrypoints**:

- **single-node MP child** — `_run_node_process` (`node.py:1038-1062`) → constructs Node,
- **process-group host** — `NodeProcess._run` / `_spawn_local` (`node_helpers.py:544-566`)
  → constructs each Node,
- **LOCAL / `--no-multiprocessing`** — `create_local` → constructs Node in the **manager**
  process. The data server then binds a *second* ephemeral port in the manager process,
  distinct from the bridge TCPSite **and** node_log's server (three servers in one
  process). `:0` makes collisions impossible; Step 9 adds an explicit "3 servers, no
  clash" assertion.

> Do **not** add registration calls to `_run_node_process` / `_spawn_local` /
> `create_local` — that would double-register. The "three contexts" note is only about
> *verifying the bind succeeds*, not three call sites.

### 4.8 Backpressure / latest-wins

Per-connection mailbox holds **one** frame; `publish` overwrites (drop-oldest), matching
iceoryx2 `latest_wins=True`. A slow browser never stalls `_processing_loop` (the loop
does a pointer-swap under each mailbox's short lock) and never grows memory (one bounded
immutable `bytes` per connection).

---

## 5. Production Gating (browser viewer makes the node produce)

`_processing_loop` skips slots with no subscribers
(`node.py:731: if slot.subscriber_count == 0: continue`). `subscriber_count` is bumped
**only** by `REGISTER_SUBSCRIBER` ctrl messages (messaging thread). A browser viewer with
no node consumer would otherwise get **zero frames** — the single easiest bug to ship.

### Decision: a **separate `viewer_count`**, OR-gated (do NOT reuse `subscriber_count`)

#### 5.1 `OutputSlot` field additions (PICKLE-SAFE)

`OutputSlot` is pickled via `cls._configure()` **before any wiring** (see its docstring,
`node_helpers.py:173-175`: *"Always empty at pickle time … so no pickle hooks needed"*),
and a populated `OutputSlot` is shipped to spawned child processes. **A
`threading.Lock` stored as a field cannot be pickled** — `default_factory` does not confer
picklability on the constructed Lock, so a `Lock` field would raise
`TypeError: cannot pickle _thread.lock` on every MP spawn.

**Resolution: lazy lock, created on first use, never a dataclass field.** Add only a
plain `int` field plus a lazily-created lock accessed through a property:

```python
@dataclass
class OutputSlot:
    dtype: DataType
    subscriber_count: int = 0
    # Browser-viewer count. Written ONLY by node_data server threads (under the
    # lazy viewer_lock); read ONLY by the processing loop. Plain int ⇒ picklable;
    # 0 at pickle time.
    viewer_count: int = field(default=0, repr=False, compare=False)
    publishers: List[Publisher] = field(default_factory=list, repr=False, compare=False)
    notifiers: List[object] = field(default_factory=list, repr=False, compare=False)
    has_ipc: bool = field(default=False, repr=False, compare=False)
    has_thread: bool = field(default=False, repr=False, compare=False)

    # Lazy lock — NOT a field, so it never participates in pickle/equality.
    @property
    def viewer_lock(self) -> "threading.Lock":
        lk = self.__dict__.get("_viewer_lock")
        if lk is None:
            lk = self.__dict__["_viewer_lock"] = threading.Lock()
        return lk
```

> Lazy creation is race-tolerant in practice: the data server always creates the lock from
> a single thread on the first viewer upgrade for that slot before any concurrent writer
> can exist; subsequent threads read the already-set `__dict__` entry. (If paranoia is
> warranted, guard the first creation with a module-level `threading.Lock()` in
> `node_data`.) The processing loop never touches `viewer_lock` — it only reads
> `viewer_count`.

#### 5.2 Gate change (`node.py:731`)

```python
for slot_name, slot in self.output_slots.items():
    if slot.subscriber_count == 0 and slot.viewer_count == 0:
        continue
    ...
```

#### 5.3 Skip the full-Data encode for viewer-only slots (MUST)

Today `prepare_encode(data)` (`node.py:749`) and the publisher loop (`node.py:754-759`)
run **unconditionally** once a slot passes the gate. With the OR-gate, a viewer-only slot
(`subscriber_count == 0`, empty `slot.publishers`) would still run `prepare_encode` on the
full 10.6 MB `Data` every tick — **defeating the reduction goal**. Guard the full encode
on having real subscribers:

```python
data = Data(slot.dtype, value[0], value[1])          # node.py:737, unchanged

# Node↔node iceoryx2 fan-out: ONLY when real node consumers exist. UNCHANGED behavior,
# now explicitly gated so viewer-only slots never pay the full-Data encode.
if slot.subscriber_count > 0:
    size, meta_bytes = prepare_encode(data)          # node.py:749
    for pub, notif in zip(slot.publishers, slot.notifiers):
        loan = pub.loan(size)
        encode_data_into(data, loan.buffer, meta_bytes=meta_bytes)
        loan.send(); notif.notify()                  # node.py:754-759, unchanged

# Browser reduced fan-out: ONLY when a viewer is attached. SEPARATE Data object.
if slot.viewer_count > 0:
    try:
        spec = node_data.viewspec_for(self.node_id, slot_name)
        reduced = reduce_for_view(data, spec)        # fail-open; never mutates `data`
        buf = encode_data(reduced)                   # eager bytes, codec.py unchanged
        node_data.publish(self.node_id, slot_name, buf)
    except Exception:
        # Browser-path failure is swallowed (printed → captured by node_log SSE).
        # Do NOT call _report_error and do NOT set tick_error — a view-only failure
        # must never mark the node errored or interrupt node↔node delivery.
        print(traceback.format_exc())
```

> Keep the existing `try/except` structure around `Data(...)` (`node.py:736-741`) intact.
> The two fan-outs above replace the single unconditional encode+publish block. The
> `tick_error`/`_clear_error_if_healthy` accounting (`node.py:721, 764-767`) is driven
> **only** by the node↔node path and the `Data(...)` construction, never by the browser
> branch.

### 5.4 Race-safe bookkeeping (the exact guards)

- **`subscriber_count`** — single writer (messaging loop, `node.py:512-521`). **Unchanged.**
- **`viewer_count`** — written **only** by `node_data` handler threads, always under
  `slot.viewer_lock`: `+1` on WS upgrade, `max(0, -1)` in a **`finally`** around the
  handler (so abrupt `BrokenPipe`/`ConnectionReset` still decrements — a leaked count keeps
  a slot producing forever).
- The **processing loop only READS** both counters in the gate. An `int` read is atomic
  under the GIL (a stale read costs/saves at most one tick) — no lock on the read side.
- **Cross-thread ownership invariant** (state it explicitly): `node.output_slots` is the
  *same* dict object the processing loop iterates (`node.py:730`), and the data server
  reaches the *same* `OutputSlot` instances via `_nodes[node_id].output_slots[slot]`. The
  data-server thread mutates `viewer_count`; the node thread reads it; same process in all
  three host contexts. This in-process attribute mutation is the only cross-thread channel
  for gating — no iceoryx2/ctrl message is involved.
- **Idle-leaf wake.** On the `0 → 1` `viewer_count` transition, the handler calls
  `node._wake_processing()` (`node.py:312`, which sends on the in-process self-trigger pub,
  `node.py:179-184`) so a node currently blocked in `self._waitset.wait(...)` ticks and
  starts producing. **Scope limit (documented, not a bug):** `viewer_count` does **not**
  cascade upstream the way `add_link`'s `REGISTER_SUBSCRIBER` chain does. Viewing a slot
  whose producer is purely *input-triggered* by an upstream that has **no** subscribers/
  viewers will not start that upstream. Supported case: viewing a free-running or already-
  ticking node. This matches the existing iceoryx2 limitation and is acceptable for the
  target workloads (sources free-run; `test.gfi`'s Oscillator/VideoStream autotrigger).

Because the browser path never calls `_ensure_output_endpoints`, **no unused iceoryx2
publisher** is provisioned for browser-only slots (`slot.publishers` stays empty → the
node↔node loop is a no-op). On Windows this also avoids needlessly touching the iox2
runtime dirs.

---

## 6. Reduction (node-side thalamus) — `src/goofi/node_reduce.py` (NEW)

### 6.1 ViewSpec (Python) + constructor

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from goofi.data import Data, DataType

@dataclass(frozen=True)
class ViewSpec:
    mode: str = "raw"                 # line|image|trajectory|topomap|string|table|scalar|raw
    max_samples: int = 2048           # line/trajectory target along `axis`
    max_channels: int = 64            # line: cap leading channel dim
    max_pixels: Tuple[int, int] = (1280, 720)   # image target (w, h)
    axis: int = -1                    # reduced sample axis for line
    version: int = 0                  # client ordering only; node ignores

def viewspec_from_dict(d: dict) -> ViewSpec:
    def _i(k, dv):
        try: return max(1, int(d.get(k, dv)))
        except Exception: return dv
    mp = d.get("max_pixels", (1280, 720))
    try: mp = (max(1, int(mp[0])), max(1, int(mp[1])))
    except Exception: mp = (1280, 720)
    mode = d.get("mode", "raw")
    if mode not in ("line","image","trajectory","topomap","string","table","scalar","raw"):
        mode = "raw"
    try: ax = int(d.get("axis", -1))
    except Exception: ax = -1
    return ViewSpec(mode=mode, max_samples=_i("max_samples", 2048),
                    max_channels=_i("max_channels", 64), max_pixels=mp,
                    axis=ax, version=int(d.get("version", 0) or 0))
```

### 6.2 Entry point

```python
def reduce_for_view(data: Data, spec: Optional[ViewSpec]) -> Data:
    """Return a (possibly) smaller Data for `spec`.

    INVARIANTS:
      * FAIL-OPEN: any guard trip or exception returns `data` UNREDUCED.
      * NEVER mutates `data` (node↔node publishers still encode the full object).
      * Every produced array is a FRESH contiguous copy (np.ascontiguousarray);
        passthrough returns `data` itself, which is safe because the caller
        encode_data(reduced) eagerly serializes to `bytes` WITHIN the same
        _processing_loop iteration, before the next tick rebuilds value[0]
        (see §6.6 lifetime invariant).
      * Co-reduces meta['channels']['dimD'] for EVERY reduced axis D with the
        SAME transform, so Data.__post_init__ (data.py:104) does not assert.
    """
    if spec is None:
        return data
    try:
        if data.dtype != DataType.ARRAY:        # STRING/TABLE → passthrough
            return data
        arr = data.data
        if spec.mode == "line":
            return _reduce_line(data, arr, spec)
        if spec.mode == "image":
            return _reduce_image(data, arr, spec)
        if spec.mode == "trajectory":
            return _reduce_trajectory(data, arr, spec)
        if spec.mode == "raw" and arr.ndim > 3:
            return _thumbnail(data, arr, spec)
        # topomap / scalar / raw-small / unknown → passthrough
        return data
    except Exception:
        return data                              # FAIL-OPEN
```

### 6.3 Reduction-policy table (dtype / viewer → algorithm → target dimension)

| `ViewSpec.mode` | Input `Data` shape | Algorithm | Target dimension |
|---|---|---|---|
| `line` | 1-D `(N,)` ARRAY | **min/max envelope** along `axis`: split `N` into `W=min(max_samples, N)` bins, emit per-bin `min,max` **interleaved** → `2*W`. **Never stride** (stride aliases away audio transients/clipping). Skip if `N < 2*W` (ratio < 2×). | `(2*W,)`; `meta['reduced']={mode:'envelope', orig:N, axis:<resolved>}` |
| `line` | 2-D `(C,N)` ARRAY | (1) cap `C`: if `C > max_channels`, gather `ci = unique-preserving np.linspace(0,C-1,max_channels).round().astype(int)` → `C'`; (2) envelope each kept row along the sample axis. | `(C', 2*W)`; both dims co-reduced (§6.4); `meta['reduced']` as above |
| `image` | 3-D `(H,W,3|4)` ARRAY | **block-mean area downscale** to fit `max_pixels` preserving aspect (algorithm §6.5). **Image skip threshold is 1× (always downscale if target < source on either axis)** — NOT the 2× line rule. | `(h',w',ch)`, `h'≤max_pixels[1]`, `w'≤max_pixels[0]` |
| `trajectory` | 2-D `(N,2)` ARRAY | uniform index resample: `idx = np.linspace(0,N-1,min(N,max_samples)).round().astype(int)`; gather. | `(min(N,max_samples), 2)` |
| `topomap` | small `(C,)` ARRAY | **passthrough** (channel count already tiny) | unchanged |
| `string` | STRING | **passthrough** | unchanged |
| `table` | TABLE | **passthrough** (browser renders tree, bounds depth) | unchanged |
| `scalar` | 0-D / `(1,)` | **passthrough** | unchanged |
| `raw` / high-dim (`ndim>3`) | any | **bounded thumbnail**: flatten and take first `max_samples` (1-D copy); frontend shows a text summary (shape/dtype/stats) when no viewer fits. | `(min(size,max_samples),)`, coords dropped |

> **Line skip-if-ratio<2×** avoids pointless copies (envelope of a near-target buffer).
> **Image uses a 1× threshold** so the headline HD case (1920×1080 → 1280×720, only 1.5×/
> axis) **always downscales** — resolving the earlier contradiction where a 2× image skip
> shipped the full 6.2 MB frame.

### 6.4 Meta co-reduction (the exact rule) + safety net

`Data.__post_init__` (`data.py:101-107`) asserts, **for every axis `d` that has a coord
list**, `len(meta['channels']['dimD']) == data.shape[d]`. Therefore each reduced axis with
a coord list must be co-reduced with the **same** transform:

- **envelope** along axis `a` → reduced length is `2*W`; co-reduced coord =
  `list(np.repeat(np.asarray(coord)[bin_centers], 2))`, where `bin_centers` is the index of
  each bin's center sample. Envelope **must guarantee** this co-reduced coord (do not rely
  on the backstop) so the §8.7 band keeps its x-axis.
- **subsample** (channel cap on dim0, trajectory on dim0) → gather the **same indices**
  used on the data: `[coord[i] for i in idx]`.
- **2-D `(C,N)` line**: co-reduce **both** — `dim0` via channel-gather (`ci`), `dim1` via
  envelope-repeat. Each independently must satisfy `len == reduced.shape[d]`.

**Backstop** (last line of defense, not the primary mechanism): immediately before
constructing the reduced `Data`, drop any `meta['channels']['dimD']` whose length ≠
`reduced.shape[d]`. The `Data(...)` constructor is the final net; any residual mismatch
raises inside `reduce_for_view`'s `try`, which **fails open** to the unreduced `data`.

```python
def _coreduce_coords(meta: dict, dim: int, new_coord: Optional[list]) -> None:
    ch = meta.setdefault("channels", {})
    k = f"dim{dim}"
    if new_coord is None:
        ch.pop(k, None)                 # axis lost its meaning → drop
    else:
        ch[k] = list(new_coord)
```

### 6.5 Helper bodies (numpy-only, full)

```python
def _envelope(x: np.ndarray, axis: int, w: int):
    """1-D envelope along `axis`: returns (env[..,2*w,..], bin_centers[w] int idx)."""
    n = x.shape[axis]
    w = min(w, n)
    edges = np.linspace(0, n, w + 1).astype(int)
    xs = np.moveaxis(x, axis, -1)           # sample axis last
    out = np.empty(xs.shape[:-1] + (2 * w,), dtype=xs.dtype)
    centers = np.empty(w, dtype=int)
    for b in range(w):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        seg = xs[..., lo:hi]
        out[..., 2 * b]     = seg.min(axis=-1)
        out[..., 2 * b + 1] = seg.max(axis=-1)
        centers[b] = (lo + hi - 1) // 2
    return np.ascontiguousarray(np.moveaxis(out, -1, axis)), centers

def _area_downscale(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Block-mean area downscale (H,W,C)->(out_h,out_w,C), numpy only.
    np.add.reduceat over integer bin edges handles non-divisor ratios."""
    H, W = img.shape[0], img.shape[1]
    out_h = min(out_h, H); out_w = min(out_w, W)
    ye = np.linspace(0, H, out_h + 1).astype(int)
    xe = np.linspace(0, W, out_w + 1).astype(int)
    f = img.astype(np.float32)
    rows = np.add.reduceat(f, ye[:-1], axis=0) / np.maximum(1, np.diff(ye))[:, None, None]
    cols = np.add.reduceat(rows, xe[:-1], axis=1) / np.maximum(1, np.diff(xe))[None, :, None]
    return np.ascontiguousarray(cols.astype(img.dtype))

def _resample_path(xy: np.ndarray, n: int):
    N = xy.shape[0]
    idx = np.linspace(0, N - 1, min(N, n)).round().astype(int)
    return np.ascontiguousarray(xy[idx]), idx
```

`_reduce_line` / `_reduce_image` / `_reduce_trajectory` / `_thumbnail` build a **new**
`meta` dict (shallow copy of `data.meta`, deep-copy the `channels` sub-dict before edits),
call the helper, set `meta['reduced']` for envelope, co-reduce coords (§6.4), and return
`Data(data.dtype, new_arr, new_meta)`. They operate on copies and never touch `data.data`.

### 6.6 Lifetime invariant (the load-bearing guarantee — replaces "never crosses a thread boundary")

`reduce_for_view(data, spec)` **and** `encode_data(reduced)` both execute **synchronously
within the same `_processing_loop` iteration** that built `data` (`node.py:737`). Only the
resulting **immutable `bytes`** are stored in the mailbox(es). Therefore:

- Passthrough returning `data` itself is safe: `encode_data` copies into `bytes` **before**
  the next tick rebuilds `value[0]`/`Data(...)`.
- No reducer output that aliases `data.data` ever outlives the tick.
- Reduction is **coupled to the tick** (like `prepare_encode`), giving automatic
  backpressure: a slow reducer simply slows the tick.

> **Acknowledged regression, accepted (§13 reconciliation).** Because reduction runs inline
> on the processing thread, viewing a slot **may slow that node's tick** by the reduce+
> encode cost, which in turn lowers its node↔node output rate *while a human is watching*.
> The node↔node **path** (its encode/SHM/transport, `codec.py`) is byte-for-byte unchanged;
> only the *tick cadence* can drop under an attached viewer. This is the chosen trade-off:
> inline coupling buys the bounded-memory, single-numpy-lifetime guarantee without a
> second reducer thread. Mitigations: line/audio envelope is ~1300× and negligible; the
> heavy case (HD downscale) is bounded by `max_pixels` and runs at most once per folded
> spec per tick. A unit test asserts (via `np.shares_memory`) that no reducer return value
> shares a buffer with `data.data`.

---

## 7. Negotiation Protocol

### 7.1 Endpoint advertising via `STATE_UPDATE` / snapshot

- **Mint** (`node.py:__init__`, ONE line beside `node.py:173-174`):
  ```python
  self._data_endpoint: Optional[str] = None        # init near node.py:150 beside _log_endpoint
  if capture_logs:                                  # same non-headless gate as logs
      self._log_endpoint  = node_log.register_node(node_id)
      self._data_endpoint = node_data.register_node(node_id, self)  # ws://127.0.0.1:<port>/<node_id>
  ```
- **Advertise** (`node.py:_push_state`, beside `node.py:423`):
  ```python
  state = { ..., "log_endpoint": self._log_endpoint,
                 "data_endpoint": self._data_endpoint }
  ```
  The endpoint is **static**, so it rides the **first** post-setup `_push_state` (which
  always fires because `setup()` marks the node dirty). Invariant: *a node always advertises
  `data_endpoint` on its first post-setup state push* — identical dependency to
  `log_endpoint`, which already works.
- **Relay** (`bridge/control.py:on_state`, beside `control.py:343`):
  ```python
  "payload": { ..., "log_endpoint":  message.content.get("log_endpoint"),
                    "data_endpoint": message.content.get("data_endpoint") }
  ```
- **Snapshot** (`bridge/schemas.py:describe_node_instance`, beside `schemas.py:99`):
  ```python
  "data_endpoint": (ref.serialized_state or {}).get("data_endpoint"),
  ```

> One **per-process base URL** (not a per-slot map): the port is per-process, so all slots
> share it; the browser appends `/<node>/<slot>`. This keeps `data_endpoint` plumbing
> byte-identical to `log_endpoint` (a single scalar `!== undefined` merge) and survives
> dynamic `output_slots` re-declaration without churn.

### 7.2 The data WS messages

| dir | when | payload |
|---|---|---|
| C→S | connect URL query (zero-RTT seed) | `<base>/<node>/<slot>?spec=<base64url(JSON ViewSpec)>` (≤ `_SEED_MAX=4096` chars; oversize ⇒ omit, rely on the post-connect message) |
| C→S | WS **TEXT** on every (re)connect and renegotiation | `{ "op":"view", "spec": <ViewSpec JSON>, "v": <int> }` |
| S→C | WS **BINARY** | a GOOF-encoded **reduced** `Data` frame (`$lib/codec/decode` unchanged) |
| C↔S | keepalive / liveness | WS `ping`/`pong` |

- **First-frame seed**: the `?spec=` query sets `_specs[(node,slot)]` **before** the first
  tick, so the very first reduced frame is correctly sized (critical — one unreduced
  44.1 kHz frame is 10.6 MB). Base64url alphabet, `=` padding restored on decode (§4.4).
- **Versioning**: `v` is **client-side ordering only**; the **server ignores it** and
  applies last-received-wins.
- **Reconnect**: no server resume. The client re-resolves `data_endpoint` from graph state,
  re-opens, re-sends the seed (query) **and** an immediate `{op:"view"}`. The server resets
  `viewer_count` correctly across reconnect (decrement-in-`finally` on close, increment on
  new upgrade), and the new connection gets its **own** mailbox.

---

## 8. Frontend Changes

### 8.0 Ownership split (RESOLVED)

- **`dataStream.svelte.ts` OWNS the per-`(node,slot)` WebSocket lifecycle** (open / close /
  move), exactly as `logStream.svelte.ts` owns its EventSources. It resolves
  `data_endpoint` from graph state and reconciles sockets against the active need set.
- **`data.ts` is repurposed to a stateless connection helper** invoked *by* `dataStream`:
  it builds the URL from a passed-in `data_endpoint` + slot + seed spec, opens the socket,
  decodes frames, exposes an `updateSpec(spec)` method to push `{op:"view"}` on an already-
  open socket, and fans decoded frames to listeners. It no longer reads `location.host`,
  no longer owns refcount/dedup of the *need* set (that moves to `dataStream`), but **keeps
  per-socket listener fan-out** so multiple `ViewerFeed`s on one `(node,slot)` share one
  socket. The previous `_subs` refcount map is folded into `dataStream`'s reconcile.

This removes the "two owners for one socket" contradiction: **`dataStream` owns sockets;
`data.ts` is its transport primitive.**

### 8.1 New: `frontend/src/lib/stores/dataStream.svelte.ts`

Mirror `logStream.svelte.ts`: a process-wide singleton with a single `$effect.root`
`reconcile()` that:

1. reads `graph().nodes` → `endpoints = Map<nodeName, data_endpoint|null>`,
2. reads the active need set `(node, slot)` (populated by `ViewerFeed` via `setNeed`/
   `release`, like logStream's `setNeeds`/`release`, mutations wrapped in `untrack`),
3. for each need whose `data_endpoint` is **known**, ensures an open connection
   (`data.ts.connect(...)`),
4. closes any connection no longer needed **or** whose
   `endpoints.get(name) !== conn.endpoint` (endpoint **moved** — node restarted on a new
   port), mirroring `logStream.svelte.ts:84`,
5. subscribes to `thalamus.viewSpecFor(node, slot)` changes and calls
   `conn.updateSpec(spec)` to push `{op:"view"}` on the live socket.

Defers opening until `data_endpoint` is known (null until first `STATE_UPDATE`).
Reconnect/backoff lives in `data.ts` but **re-resolves the endpoint from `dataStream` on
each retry** (never a cached/stale URL) — `data.ts.connect` is given a `() => endpoint`
resolver, not a frozen string. `dataStream` exposes a `(node,slot)`-keyed decoded-frame
source that `frames.ts` consumes (so `frames.ts` never sees the endpoint).

### 8.2 New: `frontend/src/lib/stores/thalamus.svelte.ts`

```ts
export interface ViewSpec {
  mode: 'line'|'image'|'trajectory'|'topomap'|'string'|'table'|'scalar'|'raw';
  max_samples: number; max_channels: number;
  max_pixels: [number, number]; axis: number; version: number;
}
// addConsumer(node, slot, id: symbol, spec) / updateConsumer / removeConsumer
// viewSpecFor(node, slot): ViewSpec   // folds all live consumers → ONE spec
```

**Folding = largest consumer wins**: `max_samples`/`max_channels` = max over consumers;
`max_pixels` = element-wise max; **mode precedence** `image > line > trajectory > topomap >
raw > scalar > string > table`. Bump `version` on each effective change. **Debounce
150 ms** + **hysteresis**: round width up to 64 px; renegotiate only on **> 25 % grow** or
**> 50 % shrink**. **Capacity math** lives here: line `max_samples =
max(64, ceil(canvasCssPx * devicePixelRatio))`; image `max_pixels = [max(1, w*dpr),
max(1, h*dpr)]`. At 0 px, clamp to the 64 floor and **do not register a need** until the
element has area.

### 8.3 Edited: `frontend/src/lib/api/data.ts`

- New shape: `connect(resolveEndpoint: () => string|null, node, slot, getSpec: () => ViewSpec)`
  returns `{ onFrame(cb), updateSpec(spec), close() }`.
- URL: `${endpoint}/${encodeURIComponent(slot)}?spec=${b64urlSpec}` (seed from `getSpec()`).
  Build base64url from `JSON.stringify(spec)`; if the encoded query exceeds 4096 chars,
  **omit** `?spec=` and rely on the post-open `{op:"view"}`.
- On `open`: send `{op:"view", spec: getSpec(), v}` immediately; on every `updateSpec`,
  send again with an incremented `v`.
- On reconnect (existing 250 ms→5000 ms backoff): call `resolveEndpoint()` again; if it
  returns a different/null endpoint, stop and let `dataStream` re-drive. Re-send seed +
  `{op:"view"}` on the new socket.
- `binaryType = 'arraybuffer'`; `decodeData(e.data)` **unchanged** (reduced frames are the
  same GOOF format, just smaller).

### 8.4 Edited: `frontend/src/lib/api/control.ts`

Add `data_endpoint?: string | null` to `NodeInstanceInfo` (beside `log_endpoint`, ~`:38`)
and to the `state_update` `ControlEvent` payload (~`:103`).

### 8.5 Edited: `frontend/src/lib/stores/graph.svelte.ts`

In the `state_update` handler (beside the `log_endpoint` merge, ~`:191`):
```ts
if (ev.payload.data_endpoint !== undefined) t.data_endpoint = ev.payload.data_endpoint;
```
Add `data_endpoint?: string | null` to the node type. `hello` / `node_added` already carry
it via `NodeInstanceInfo`.

### 8.6 Edited: `frontend/src/lib/viewers/ViewerFeed.svelte`

At the `visible` toggle (IntersectionObserver, ~`ViewerFeed.svelte:27-44`): mint
`const consumerId = Symbol()`, observe the canvas via `ResizeObserver`, and
`thalamus.addConsumer/updateConsumer/removeConsumer` on the `visible && expanded` effect.
Register/release the `(node, slot)` **need** with `dataStream` on the same effect.
`frames.ts` (RAF coalesce to 1 frame/paint) and `subscribeFrames(node, slot, cb)` stay
**unchanged in signature** — they call into `dataStream`'s per-`(node,slot)` frame stream,
so the endpoint/ViewSpec machinery is entirely upstream of `frames.ts`.

> The earlier "endpoint move lives entirely inside data.ts while frames.ts is unchanged"
> claim is corrected: the move lives in **`dataStream`**; `frames.ts` keeps its
> `(node, slot)` signature because `dataStream` presents a `(node, slot)`-keyed frame
> source. `frames.ts` source code is unchanged.

### 8.7 ArrayViewer envelope band

`frontend/src/lib/codec/decode.ts` **already exposes `meta`** on `DataFrame`
(`decode.ts:40`: `meta: Record<string, unknown>`), so `meta.reduced` survives decode with
**no decode change**. Add a typed accessor in the viewer:

```ts
const reduced = (f.meta.reduced ?? null) as { mode?: string; orig?: number; axis?: number } | null;
```

In `frontend/src/lib/viewers/ArrayViewer.svelte`: when
`reduced?.mode === 'envelope'`, the body length along the reduced axis is `2*W` interleaved
`min,max`. **De-interleave** in the viewer: `mins = values[0::2]`, `maxs = values[1::2]`,
and render a **min/max band** (uPlot fill between the two series) instead of a single line,
so audio transients/clipping stay visible. For non-envelope frames, render the single
series as today.

---

## 9. Manager / Bridge Changes + Removals

### 9.1 DELETE (manager-side data path)

- **`src/goofi/bridge/data.py`** — entire file (`DataHub`, `_SlotForwarder`, `on_frame`).
  It is the **only** caller of `NodeRef.set_data_handler`.
- **`src/goofi/bridge/server.py`**:
  - drop `from goofi.bridge.data import DataHub` (`server.py:24`),
  - drop `self.data = DataHub(self)` (`server.py:165`),
  - drop the route `web.get("/data/{node}/{slot}", self.data.handler)` (`server.py:209`),
  - drop `await self.data.close_all()` in shutdown (`server.py:268`).
- **`src/goofi/node_helpers.py`** (dearpygui-era manager-decode hook):
  - `NodeRef.set_data_handler` (`node_helpers.py:400-435`),
  - `NodeRef._data_pump` (`node_helpers.py:437-473`),
  - `NodeRef.open_output_subscriber` (`node_helpers.py:392-398`) and
    `NodeRef.data_service_for` (`node_helpers.py:389-390`, only used by it),
  - `__post_init__` data-pump fields (`_data_handlers`, `_data_handlers_lock`,
    `_data_waitset`, `_data_waitset_dirty`, `_data_pump_thread`, ~`node_helpers.py:240-244`)
    and the `_data_waitset_dirty.set()` in `terminate` (~`node_helpers.py:371`),
  - the now-unused `WaitSet` import if nothing else in the file uses it (grep before
    deleting the import).

### 9.2 KEEP (control plane + graph ownership)

- `/control` WS (`bridge/control.py`), RPC dispatch, `_snapshot`, node/link add/remove
  events, `_wire_node_status` STATE_UPDATE + PROCESSING_ERROR relay, and the **log SSE**
  relay (`control.py:343`).
- `Manager` graph + link ownership (`NodeContainer`, `_links`, `add_node`/`remove_node`/
  `add_link`/`remove_link`/`save`/`load`).
- `NodeRef.register_subscriber`/`unregister_subscriber` and their ctrl handling
  (`node.py:512-521`) — **KEEP**: used by real node↔node links and asserted by tests
  (`tests/test_node.py:136-148`, `tests/test_manager.py:74`).
- Static SPA serving + SPA fallback + `no_cache` middleware.

### 9.3 Confirm UNTOUCHED

- **Node↔node iceoryx2 publish** (`node.py:754-759`, now guarded by `subscriber_count > 0`
  but byte-for-byte identical work when subscribers exist), `_ensure_output_endpoints`,
  `SUBSCRIBE_INPUT` wiring, `transport.py`, and **`codec.py`** — unchanged. The reduced
  browser frame is a **separate** encode of a **separate** reduced `Data`; the full `Data`
  still flows to iceoryx2 publishers (CLAUDE.md §13). `frames.ts` and
  `frontend/src/lib/codec/decode.ts` — unchanged.

---

## 10. Edge Cases and Risks

- **Process-group host**: one process, many nodes, **one port**, keyed by `(node_id,
  slot)`. One server, never one-per-node. No thread-attribution (explicit `publish`).
- **LOCAL / `--no-multiprocessing`**: node + data server + log server + bridge TCPSite all
  in the manager process on distinct `:0` ports. Loopback-P2P, manager not transcoding.
  Step 9 asserts no clash.
- **Multi-viewer dedup (one tab)**: two `ViewerFeed`s of one slot → one need → one socket
  (`dataStream`) → one folded ViewSpec → node reduces once → fanned to the one connection
  mailbox → `data.ts` fans to both listeners.
- **Multi-tab / overlapping reconnect**: each tab/socket gets its **own** connection
  mailbox; `publish` fans the same bytes to all → no frame-stealing. `viewer_count` counts
  each connection; last close (incl. abrupt, via `finally`) returns it to 0.
- **Reconnect**: WS has no resume; `data.ts` backoff re-resolves the endpoint and re-sends
  the ViewSpec. Reader thread is unblocked by `socket.shutdown(SHUT_RDWR)` on handler exit
  → no thread leak.
- **Node restart**: new ephemeral port ⇒ `data_endpoint` changes ⇒ `dataStream` reconcile
  closes the stale socket and reopens against the new URL.
- **Dtype change mid-stream**: each frame self-describes via the GOOF header/meta;
  `reduce_for_view` dispatches per `ViewSpec.mode` + actual shape, fail-open on mismatch
  (e.g. an image arriving on a line-mode slot → passthrough). Viewer reacts on decode.
- **0 px layout**: gated on `visible && expanded`; thalamus clamps to a 64 floor and the
  consumer registers **no need** until it has area.
- **HiDPI**: capacity uses `devicePixelRatio`; hysteresis prevents renegotiation thrash.
- **Backpressure**: per-connection latest-wins mailbox (drop-oldest), matching iceoryx2.
  Inline reduction on the tick is the natural rate limiter (accepted §6.6 trade-off).
- **CORS / remote browser**: node servers send `Access-Control-Allow-Origin: *`, bind
  `127.0.0.1` only. **Limitation** (identical to `log_endpoint`): if `--bind` exposes the
  manager on a LAN, node-side `127.0.0.1` data endpoints are **not reachable** from a
  remote browser. Document/accept; single-user local app.
- **Risks (with mitigations)**: (1) forgetting the OR-gate ⇒ silent no-frames — Step 4
  acceptance asserts a frame arrives via `node_data.publish`; (2) leaked `viewer_count` on
  abrupt close ⇒ decrement-in-`finally` + a Step 9 leak test; (3) reducer mutating node↔node
  `Data` ⇒ §6.6 copy/passthrough invariant + a `shares_memory` test; (4) meta co-reduction
  miss ⇒ envelope guarantees the coord, backstop drops mismatches, `Data(...)` is the net,
  fail-open; (5) hand-rolled RFC6455 (unmask, 64-bit length, single write-lock, reader-
  teardown) — isolate + unit-test at 100 B/70 KB/3 MB; (6) `_reset_for_tests` parity or the
  128-test suite flakes on leaked threads/ports; (7) Lock-as-field pickle crash ⇒ lazy
  `viewer_lock` property, never a dataclass field.

---

## 11. Ordered, Commit-Sized Implementation Plan

> Backend-first; each step independently testable. Run `pytest tests/` after each backend
> step (128 tests must stay green).

**Step 1 — `OutputSlot.viewer_count` (plain int) + lazy `viewer_lock` + OR-gate + split-encode.**
*What*: add `viewer_count` field and the `viewer_lock` property to `OutputSlot`
(`node_helpers.py`, ~`:170`); change the gate at `node.py:731` to
`subscriber_count == 0 and viewer_count == 0`; split the post-`Data` block so
`prepare_encode`+publisher loop is guarded on `subscriber_count > 0` (§5.3).
*Files*: `src/goofi/node_helpers.py`, `src/goofi/node.py`.
*Acceptance*: `pytest tests/` green; `pickle.dumps(OutputSlot(DataType.ARRAY))` and a
`copy.deepcopy` succeed (no Lock in pickle); a unit test sets `slot.viewer_count = 1` on a
slot with `subscriber_count == 0`, monkeypatches `node_data.viewspec_for`/`publish`, ticks
the loop, and asserts `node_data.publish` is called (NOT that `process()` ran).

**Step 2 — `node_data.py` server + framing + `_reset_for_tests`.**
*What*: new module per §4: per-process `ThreadingHTTPServer`, hand-rolled RFC6455
(handshake, client unmask, server 7/16/64-bit write via `_send_frame`, single per-conn
`_write_lock`, ping/pong/close, reader thread + `shutdown` teardown), per-connection
`_ConnMailbox` + `(node,slot)->[mailboxes]` registry, `_specs` last-received-wins,
`register_node(node_id, node)`/`unregister_node`/`viewspec_for`/`publish`,
`_reset_for_tests`. CORS `*`, `127.0.0.1:0`. Land Step 3 first or stub the `ViewSpec` import.
*Files*: `src/goofi/node_data.py`.
*Acceptance*: framing unit tests round-trip at **100 B / 70 KB / 3 MB**; a test starts the
server, registers a fake node with one output slot, opens a raw-socket WS, receives a
`publish()`ed binary frame, sends a **masked** TEXT `{op:"view",spec,v}` and asserts
`viewspec_for` updates; closing the socket decrements `viewer_count` to 0 and leaves no
live reader thread; `_reset_for_tests` frees the port (rebind `127.0.0.1:<port>` succeeds).

**Step 3 — `node_reduce.py` + ViewSpec.**
*What*: `ViewSpec` (frozen dataclass), `viewspec_from_dict`, `reduce_for_view`,
`_envelope`/`_area_downscale`/`_resample_path`/`_thumbnail`/`_coreduce_coords` per §6.
Fail-open; never mutate input; numpy-only.
*Files*: `src/goofi/node_reduce.py`.
*Acceptance*: unit tests — 1-D line → `2*W` envelope with co-reduced `dim0`
(`np.repeat(bin_centers,2)`); 2-D `(C,N)` → channel cap (dim0 gather) + envelope (dim1),
both coords correct, `Data(...)` constructs without tripping `data.py:104`; HD image
`1920×1080×3` → `≤ max_pixels` (downscales at 1.5×/axis); line skip-if-ratio<2× returns
input; a deliberately broken input returns the **unreduced** object; a `np.shares_memory`
test confirms no reduced array aliases `data.data`.

**Step 4 — Wire node-side: bootstrap, advertise, inline reduce+publish, idle-wake, teardown.**
*What*: init `self._data_endpoint = None` (~`node.py:150`); `register_node(node_id, self)`
in `__init__` (beside `node.py:173-174`); add `"data_endpoint"` to `_push_state`
(`node.py:423`); inline reduce+`encode_data`+`publish` after `node.py:737` gated on
`viewer_count > 0` (§5.3), swallowing browser-path errors; `node._wake_processing()` on the
`0→1` viewer transition (done in the data server handler, §4.3); `unregister_node` in
`_messaging_loop` teardown (beside `node.py:490-491`). **No edits** to `_run_node_process`/
`_spawn_local`/`create_local`.
*Files*: `src/goofi/node.py`.
*Acceptance*: spawn a node with capture on; assert its `STATE_UPDATE` carries
`ws://127.0.0.1:<port>/<node>` as `data_endpoint`; connect a raw WS, send a ViewSpec, assert
reduced frames arrive and shrink with `max_samples`; with a real node consumer ALSO
attached, assert node↔node iceoryx2 frames are still **full** size; viewing an idle free-
running node produces frames after connect.

**Step 5 — Bridge relay + snapshot for `data_endpoint`.**
*What*: relay in `control.py:on_state` (beside `control.py:343`); add to
`schemas.describe_node_instance` (beside `schemas.py:99`).
*Files*: `src/goofi/bridge/control.py`, `src/goofi/bridge/schemas.py`.
*Acceptance*: a `/control` test client gets `data_endpoint` in both the `hello` snapshot
and a `state_update` event.

**Step 6 — Remove the manager-side data path.**
*What*: delete `bridge/data.py`; remove `DataHub` import/instantiation/route/`close_all`
from `server.py`; delete `NodeRef.set_data_handler`/`_data_pump`/`open_output_subscriber`/
`data_service_for` + data-pump fields + `terminate`'s `_data_waitset_dirty.set()`.
*Files*: `src/goofi/bridge/server.py`, `src/goofi/bridge/data.py` (deleted),
`src/goofi/node_helpers.py`.
*Acceptance*: `pytest tests/` green; `git grep -n set_data_handler src/goofi` empty; bridge
starts and serves `/control` + static.

**Step 7 — Frontend: types + graph merge + endpoint discovery + WS ownership.**
*What*: `control.ts` `data_endpoint`; `graph.svelte.ts` merge; new `dataStream.svelte.ts`
(reconcile, endpoint-moved compare, owns sockets); refactor `data.ts` to the
`connect(resolveEndpoint, node, slot, getSpec)` helper shape (§8.3).
*Files*: `frontend/src/lib/api/control.ts`, `frontend/src/lib/stores/graph.svelte.ts`,
`frontend/src/lib/stores/dataStream.svelte.ts`, `frontend/src/lib/api/data.ts`.
*Acceptance*: `tsc` strict (no `any` in app code); a Playwright test loads a patch and
asserts the node's `data_endpoint` appears in the store after the first `state_update`, and
a WS to `127.0.0.1:<nodeport>` opens (DevTools network).

**Step 8 — Frontend: thalamus store + inband ViewSpec + ViewerFeed lifecycle + ArrayViewer band.**
*What*: `thalamus.svelte.ts` (fold/debounce/hysteresis/capacity); `ViewerFeed` consumer +
need lifecycle; `dataStream` pushes folded spec via `data.ts.updateSpec`; ArrayViewer
de-interleaves envelope and renders a min/max band.
*Files*: `frontend/src/lib/stores/thalamus.svelte.ts`,
`frontend/src/lib/viewers/ViewerFeed.svelte`, `frontend/src/lib/viewers/ArrayViewer.svelte`,
`frontend/src/lib/stores/dataStream.svelte.ts` (spec push).
*Acceptance*: `e2e/viewers.spec.ts` — open ArrayViewer on an Oscillator/PSD slot, assert a
reduced **band** render appears and reflows on canvas resize; image viewer downscales an HD
frame; no console errors.

**Step 9 — Stress + leak + co-location tests.**
*What*: open `test.gfi` (Oscillator + PSD + 8 Buffers + VideoStream) with 10+ viewers for
60 s; a Python test asserting `viewer_count` returns to 0 after a WS abruptly closes (leak
guard) and the reader thread terminated; a LOCAL-mode test asserting the bridge TCPSite,
node_log server, and node_data server bind distinct ports without error.
*Files*: `e2e/stress.spec.ts`; `tests/test_node_data.py`.
*Acceptance*: median ≥ 55 fps, no JS console errors, no Python tracebacks; `viewer_count`
returns to 0 on abrupt disconnect; no leaked reader thread; LOCAL 3-server bind clean;
`pytest tests/` still green.

**Step 10 — Test fixture wiring.**
*What*: wherever the suite resets `node_log` (find via `git grep -n
"node_log._reset_for_tests"` — typically `tests/conftest.py`), also call
`node_data._reset_for_tests()`.
*Files*: the existing reset fixture (likely `tests/conftest.py`).
*Acceptance*: repeated full `pytest tests/` runs show no port/thread leakage.

---

## 12. What Stays Untouched / Done When

### Stays untouched (HARD)

- `src/goofi/transport.py`, `src/goofi/codec.py`, the iceoryx2 setup.
- Node↔node publish work: `node.py:754-759` (now guarded on `subscriber_count > 0`, same
  work when subscribers exist), `_ensure_output_endpoints`, `SUBSCRIBE_INPUT`.
- `_processing_loop`'s full-`Data` construction (`node.py:737`).
- `Manager` graph/link ownership, persistence, `/control` RPC + events, log SSE relay,
  static SPA serving + `no_cache` middleware.
- `NodeRef.register_subscriber`/`unregister_subscriber` and node ctrl handling
  (`node.py:512-521`).
- `frontend/src/lib/api/frames.ts` (RAF coalescer signature) and
  `frontend/src/lib/codec/decode.ts` (already exposes `meta`).

### Done when

1. A browser viewing a slot connects **directly** to
   `ws://127.0.0.1:<nodeport>/<node>/<slot>` (verified in DevTools); the **manager is not**
   in the data path (`git grep -n set_data_handler src/goofi` empty; `bridge/data.py`
   deleted).
2. The node **reduces** before encode: a 44.1 kHz line slot ships ~`2*max_samples` envelope
   frames (KB, not MB); an HD image slot ships a downscaled frame `≤ max_pixels` (the 1×
   image threshold downscales 1920×1080→1280×720).
3. `viewer_count` gates production: a browser-only slot produces; closing the last viewer
   (incl. abrupt close, via `finally`) returns `viewer_count` to 0 and the slot stops
   (unless a node consumer remains); a node consumer is never starved by a browser
   disconnect; the full-Data encode is **skipped** for viewer-only slots.
4. `data_endpoint` rides `STATE_UPDATE` + snapshot exactly like `log_endpoint`; the frontend
   discovers it, reconnects across node restart (new port), and re-sends its ViewSpec.
5. Reduction is **fail-open**, **meta co-reduces** (no `data.py:104` assertions), and never
   mutates or aliases the node↔node `Data` beyond the tick (`shares_memory` test).
6. `pytest tests/` (128 tests) green; RFC6455 framing unit tests pass at 100 B/70 KB/3 MB;
   `e2e/viewers.spec.ts` + `e2e/stress.spec.ts` pass (≥ 55 fps median, no console errors).
7. `node_data._reset_for_tests` parity — no leaked server/reader threads or bound ports
   across the suite; LOCAL-mode 3-server bind verified clean.