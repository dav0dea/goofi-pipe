# goofi-pipe Peer-to-Peer Viewer-Data Plane with Node-Side Thalamus Reduction

> **Revision 2026-06-16** — applies three user-directed changes on top of the original
> peer-to-peer / thalamus design:
> - **Change A** — reduction runs on a **dedicated per-process reducer thread**, not inline
>   on `_processing_loop`. The node tick (and thus its node↔node output rate) no longer pays
>   the reduce+encode cost while a viewer is attached.
> - **Change B** — node log **and** data servers share the **frontend's host scope**: the
>   manager exports `GOOFI_BIND_HOST` (default `0.0.0.0`, LAN-reachable) before any node
>   spawn; both node servers bind it; `register_node` advertises a **port**; the frontend
>   composes URLs from `location.hostname`. The old "127.0.0.1 only / no remote browser"
>   limitation is removed and replaced with a LAN security note.
> - **Change C** — the ViewSpec is **per-axis** (`{axes:[{axis,max,method}], version}`);
>   `reduce_for_view` composes per-axis reductions; the folded spec is per-axis; and the
>   **meta inspector is reduction-aware** — it reconstructs and shows the **true original
>   meta** off the single per-slot reduced stream.
>
> Everything else (RFC6455 framing, pickle-safe `viewer_count` + lazy `viewer_lock`,
> per-connection mailboxes + publish fan-out, fail-open, negotiation seed/versioning,
> node↔node iceoryx2 + `codec.py` untouched) is preserved verbatim.

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
  process), advertises its port as `data_port` via `STATE_UPDATE`.
- The browser discovers `data_port` from the control plane (like `log_port`), composes the
  URL from `location.hostname`, connects **directly** to the node process, and sends a
  **ViewSpec** describing what it can display (per-axis).
- A **dedicated reducer thread** in the node-host process **reduces** the live `Data` to
  the folded ViewSpec, encodes it, and streams the small reduced frames to the browser.
  The node's `_processing_loop` only **offers** the live `Data` to the reducer (O(1),
  pointer-swap + notify) — it never reduces or encodes inline (Change A).
- The manager only relays the endpoint port. The node↔node iceoryx2 path and
  `codec.py` are **untouched**.

---

## 2. Target Architecture (ASCII)

```
                            ┌──────────────────────── browser tab ───────────────────────┐
                            │  SvelteKit SPA                                              │
                            │   ┌─ /control WS ──────────────► manager (graph + events)   │
                            │   │      hello / state_update {data_port, log_port}         │
                            │   │                                                          │
                            │   │  dataStream store (mirrors logStream): COMPOSES URL from │
                            │   │  location.hostname + node.data_port AND OWNS the         │
                            │   │  per-(node,slot) WS                                      │
                            │   │  thalamus store: folds ViewerFeed consumers → per-axis   │
                            │   │  ViewSpec                                                │
                            │   ▼                                                          │
                            │  per (node,slot) WS  ───────────────────────┐                │
                            └──────────────────────────────────┬──────────┼────────────────┘
                                                               │          │
   ┌──────────── manager process ────────────┐                │ (control)│ (data, P2P)
   │  Manager.__init__:                       │◄───────────────┘          │
   │    os.environ["GOOFI_BIND_HOST"]=        │                           │
   │      bridge_host (set BEFORE any spawn)  │   advertises only the     │
   │  bridge/server.py  (HTTP + /control WS)   │   data_port (int)         │
   │  bridge/control.py (RPC + state relay)    │                           │
   │  ── NO data path ──                       │                           │
   │  (DataHub / _data_pump REMOVED)           │                           │
   └───────────────────┬──────────────────────┘                           │
                       │ ctrl pub / status sub (iceoryx2)                  │
                       ▼                                                   ▼
   ┌──────────── node-host process (1..N nodes) ──────────────────────────────────────────┐
   │  node_data.py: ONE ThreadingHTTPServer @ ${GOOFI_BIND_HOST}:0 (per process)            │
   │     ws://<host>:<port>/<node_id>/<slot>  (hand-rolled RFC6455; binary DOWN,            │
   │                                          text per-axis ViewSpec UP; CORS *)            │
   │     PER-CONNECTION latest-wins byte mailbox; registry: (node,slot)->[mailboxes]        │
   │     PER-(node,slot) folded ViewSpec (last-received-wins)                               │
   │                                                                                        │
   │  ┌── REDUCER THREAD (one per host process) ──────────────────────────────────────┐    │
   │  │  loop: wait on _dirty; claim skey + pop _pending (latest-wins) under lock;      │    │
   │  │        spec=_specs.get(skey); reduced=reduce_for_view(snapshot, spec);          │    │
   │  │        buf=encode_data(reduced); publish(node,slot,buf)  # fan to mailboxes     │    │
   │  │        (reduce/encode errors: print+drop; never touch node state/node↔node)     │    │
   │  └─────────────────────────────────────────────────────────────────────────────────┘   │
   │                                  ▲ offer(node,slot,snapshot)  (O(1) pointer-swap+notify) │
   │  Node._processing_loop (node.py:730):                                               │   │
   │     if subscriber_count==0 and viewer_count==0: continue                            │   │
   │     data = Data(slot.dtype, value[0], value[1])   # node.py:737 (live, pre-encode)  │   │
   │     ├─ if subscriber_count>0:  prepare_encode(data); iceoryx2 publish FULL  ◄── UNCHANGED
   │     └─ if viewer_count>0:      node_data.offer(node, slot, data)  # NO reduce/encode here
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

> The reduce/encode **lane is the reducer thread**, not the `_processing_loop` lane. The
> processing loop's only viewer-path action is `offer()` (Change A).

---

## 3. Sequence Diagrams (ASCII)

### (a) Browser opens a viewer → discover → connect → negotiate → receive frames

```
ViewerFeed   thalamus store     dataStream(owns WS)   graph store   manager(/control)   node-host data server   reducer thread   node _processing_loop
   │ visible+expand (IO/RO)│            │                 │              │                    │                     │                   │
   ├─ addConsumer(spec) ──►│            │                 │              │                    │                     │                   │
   │                       ├─ viewSpecFor(node,slot) (fold per-axis, debounce 150ms)         │                     │                   │
   │                       │            │◄── read graph().node.data_port (null until 1st STATE_UPDATE)            │                   │
   │                       │            │                 │◄── state_update {data_port} ──────┤ (advertised)       │                   │
   │                       │            │                 ├─ merge data_port                  │                     │                   │
   │                       ├─ note need►│ open WS ws://${location.hostname}:${data_port}/<node>/<slot>?spec=<b64url>►│ upgrade 101 (CORS *)
   │                       │            │                 │              │                    ├─ viewer_count += 1 (lock)
   │                       │            │                 │              │                    ├─ +wake_processing (0→1)
   │                       │            │                 │              │                    ├─ _specs[(node,slot)] = seed
   │                       │            │ send TEXT {op:"view",spec:{axes,version}} ──────────────────────────────►│ _specs[(node,slot)] = spec (LWW)
   │                       │            │                 │              │                    │                     │                   ├─ gate: subscriber||viewer>0 ⇒ produce
   │                       │            │                 │              │                    │                     │                   ├─ subscriber>0? encode→SHM (node↔node, sync)
   │                       │            │                 │              │                    │                     │◄─ offer(node,slot,snapshot) ──┤ viewer>0 (O(1))
   │                       │            │                 │              │                    │                     ├─ pop _pending; reduce_for_view
   │                       │            │                 │              │                    │                     ├─ encode_data(reduced)
   │                       │            │                 │              │                    │◄ publish(node,slot,buf)┤ fan to each conn mailbox
   │◄── onFrame(decoded) ◄ RAF coalesce ◄ binary WS frame ◄──────────────────────────────────┤ handler writes latest │
```

### (b) Collapse / disconnect

```
ViewerFeed        thalamus/dataStream         node-host data server        reducer + processing loop
   │ hidden OR collapsed │                            │                            │
   ├─ removeConsumer ───►│                            │                            │
   │                     ├─ no more needs? close WS ─►│ socket.shutdown(SHUT_RDWR) │
   │                     │                            ├─ reader thread unblocks+exits
   │                     │                            ├─ handler loop exits         │
   │                     │                            ├─ finally: viewer_count -=1 (lock, max(0,…))
   │                     │                            │   ON 1→0 transition for this slot ALSO:
   │                     │                            │     _pending.pop(skey); _dirty.discard(skey);
   │                     │                            │     _specs.pop(skey)  (under reducer lock)
   │                     │                            ├─ drop this conn's mailbox    │
   │                     │                            │                            ├─ gate false ⇒ slot skipped (no offer)
   │                     │                            │                            │   (unless a node consumer remains)
```

### (c) Node restart / reconnect (new ephemeral port)

```
graph store          dataStream(owns WS)          old node-host (dead)     new node-host
   │ node removed/re-added │                            │                       │
   │ OR new STATE_UPDATE   │                            │                       │
   │  data_port CHANGED    │                            │                       │
   ├─ merge new port ─────►│                            │                       │
   │                       ├─ reconcile: ports.get(name) !== ws.port             │
   │                       ├─ close stale WS (was retrying dead port) ──────────►X (gone)
   │                       ├─ open WS ws://${location.hostname}:${new_port}/<node>/<slot>?spec ►│ 101
   │                       ├─ re-send {op:"view",spec:{axes,version}} (no server resume) ──────►│ _specs set (LWW)
```

> Reconnect rule: the data WS has **no resume buffer** (latest-wins, history-less). On
> every (re)connect the client **re-resolves `data_port` from graph state**, **re-composes
> the URL from `location.hostname`** (never retries a stale port), and **re-sends its
> ViewSpec** (seed query + immediate `{op:"view"}`).

---

## 4. Data-Server Design (`src/goofi/node_data.py`, NEW)

A structural twin of `src/goofi/node_log.py`, differing in: binary WebSocket transport
(not SSE text), per-`(node, slot)` granularity, an inbound per-axis ViewSpec channel,
**per-connection** mailboxes, `viewer_count` bookkeeping on the `OutputSlot`, and a
**dedicated reducer subsystem** (Change A) that decouples reduce+encode from the node tick.

### 4.0 Dependency surface (PINNED)

- `node_data.py`: **stdlib only** (`http.server`, `socket`, `threading`, `hashlib`,
  `base64`, `struct`, `json`, `os`, `urllib.parse`). **No** `websockets`, **no** `aiohttp`,
  **no** asyncio loop in node processes (heavy per-process cost; in `--no-multiprocessing`
  / LOCAL mode it would collide with the manager's own aiohttp app).
- `node_reduce.py`: **numpy only**. **No PIL, no scipy, no cv2.** All reductions are
  pure-numpy.

### 4.1 Transport: hand-rolled RFC6455 over stdlib `ThreadingHTTPServer`

- One process-global
  `ThreadingHTTPServer((os.environ.get("GOOFI_BIND_HOST", "127.0.0.1"), 0), _DataRequestHandler)`
  with `daemon_threads = True` (copy `node_log.py:236-243`). The bind host comes from
  `GOOFI_BIND_HOST` (Change B: the manager sets this to its `--bind` host, default
  `0.0.0.0`; unset ⇒ `127.0.0.1` for tests/standalone). OS-assigned ephemeral port read
  from `server_address[1]`. Thread name `"goofi-data-ws"` (distinct from log server's
  `"goofi-log-sse"`).
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
import base64, hashlib, json, os, socket, struct, threading, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse, parse_qs
from goofi.codec import encode_data
from goofi.node_reduce import ViewSpec, viewspec_from_dict, reduce_for_view

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

# ---- REDUCER SUBSYSTEM (Change A) -----------------------------------------
# Latest-wins LIVE-Data mailbox + dirty set + condition + the one reducer thread.
_reducer_cond = threading.Condition()
_pending: Dict[Tuple[str, str], "Data"] = {}   # latest live Data per dirty slot
_dirty: "collections.deque[Tuple[str, str]]" = collections.deque()  # round-robin order
_dirty_set: set = set()                         # membership guard for the deque
_reducer_stop = False
_reducer_thread: Optional[threading.Thread] = None


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
        host = os.environ.get("GOOFI_BIND_HOST", "127.0.0.1")   # Change B
        self._httpd = ThreadingHTTPServer((host, 0), _DataRequestHandler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever,
                         name="goofi-data-ws", daemon=True).start()


def register_node(node_id: str, node: "Node") -> int:
    """Ensure the per-process server + reducer thread, register the node, return the PORT.

    Change B: returns an int PORT (not a URL). The frontend composes the host from
    location.hostname. Mirrors node_log.register_node (also returns a port now).
    ONE call site: Node.__init__ (see §7.1). All three host contexts (single-node MP,
    group host, LOCAL) construct the Node, so they all hit this one line — do NOT edit
    _run_node_process / _spawn_local / create_local.
    """
    with _lock:
        _nodes[node_id] = node
        _ensure_server_locked()      # also starts the reducer thread (idempotent)
        port = _server_port
    return port


def unregister_node(node_id: str) -> None:
    """Drop the node; close+wake all its connection mailboxes (node terminated);
    evict any reducer state for the node (Change A leak guard)."""
    with _lock:
        _nodes.pop(node_id, None)
    with _mailboxes_lock:
        dead = [k for k in _mailboxes if k[0] == node_id]
        for k in dead:
            for mb in _mailboxes.pop(k):
                mb.close()
    for k in [k for k in _specs if k[0] == node_id]:
        _specs.pop(k, None)
    # Evict reducer state for this node so no live Data is pinned after teardown.
    with _reducer_cond:
        for k in [k for k in _pending if k[0] == node_id]:
            _pending.pop(k, None)
            _dirty_set.discard(k)
        # rebuild the deque without this node's keys (cheap; churn is rare)
        kept = [k for k in _dirty if k[0] != node_id]
        _dirty.clear(); _dirty.extend(kept)


def evict_slot(node_id: str, slot: str) -> None:
    """Called on the viewer_count 1->0 transition for a slot: release any pinned
    live Data + folded spec so latest-wins memory bound is 'one Data per CURRENTLY-
    viewed slot' (not 'per ever-viewed slot'). (Change A leak fix.)"""
    skey = (node_id, slot)
    with _reducer_cond:
        _pending.pop(skey, None)
        _dirty_set.discard(skey)
        if skey in _dirty:
            try: _dirty.remove(skey)
            except ValueError: pass
    _specs.pop(skey, None)


def viewspec_for(node_id: str, slot: str) -> Optional[ViewSpec]:
    return _specs.get((node_id, slot))           # atomic dict read under GIL


def offer(node_id: str, slot: str, data: "Data") -> None:
    """Called from the node's _processing_loop thread when viewer_count>0.

    O(1) handoff: SNAPSHOT the Data on THIS (node) thread — a contiguous COPY of the
    array + a copied meta — so the reducer only ever touches a private copy, never live
    node state (see §6.6). NO reduce, NO encode here. Pointer-swap + notify under a short
    lock; the loop never blocks. The snapshot memcpy is serialized with the node's own
    mutations (same thread), so there is no cross-thread aliasing or torn read.
    """
    snap = _snapshot_for_offer(data)             # meta defensive copy (§6.6)
    skey = (node_id, slot)
    with _reducer_cond:
        _pending[skey] = snap                    # overwrite ⇒ latest-wins drop
        if skey not in _dirty_set:
            _dirty_set.add(skey)
            _dirty.append(skey)                  # round-robin fairness (deque, not set.pop)
        _reducer_cond.notify()


def publish(node_id: str, slot: str, frame: bytes) -> None:
    """Called from the REDUCER thread. Fan immutable bytes to every live connection
    mailbox for this slot (pointer-swap + notify)."""
    with _mailboxes_lock:
        boxes = list(_mailboxes.get((node_id, slot), ()))
    for mb in boxes:
        mb.push(frame)


def _reducer_loop() -> None:
    """The ONE reducer thread per host process. Claims a dirty slot, reduces its
    latest pending Data, encodes, publishes. Predicate-loop (no missed wakeup); a
    re-offer during reduce re-adds the slot so the next-latest is reduced too.
    Round-robin over the dirty deque so no slot starves another."""
    while True:
        with _reducer_cond:
            while not _dirty and not _reducer_stop:
                _reducer_cond.wait()
            if _reducer_stop:
                return
            skey = _dirty.popleft()              # round-robin claim
            _dirty_set.discard(skey)
            data = _pending.pop(skey, None)      # claim BOTH atomically under lock
            spec = _specs.get(skey)
        if data is None:
            continue
        try:
            reduced = reduce_for_view(data, spec)   # fail-open; fresh array
            buf = encode_data(reduced)              # codec.py unchanged
            publish(skey[0], skey[1], buf)
        except Exception:
            # Fail-open: print (→ node_log SSE) + drop this frame. NEVER touches node
            # state, tick_error, or node↔node. Prefix with (node,slot) for diagnosis
            # since the reducer thread serves many nodes and is not node-attributed.
            print(f"[node_data reducer] reduce/encode failed for "
                  f"{skey[0]}/{skey[1]}:\n{traceback.format_exc()}")


def _ensure_server_locked() -> None:
    global _server, _server_port, _reducer_thread, _reducer_stop
    if _server is None:
        _server = _DataServer()
        _server_port = _server.port
    if _reducer_thread is None or not _reducer_thread.is_alive():
        _reducer_stop = False
        _reducer_thread = threading.Thread(
            target=_reducer_loop, name="goofi-data-reducer", daemon=True)
        _reducer_thread.start()


def _reset_for_tests() -> None:
    """Test-only: stop+join the reducer thread, shut the server, free the port,
    clear ALL globals incl. _pending/_dirty (mirror node_log)."""
    global _server, _server_port, _reducer_thread, _reducer_stop
    # Stop the reducer first so it cannot re-touch state mid-teardown.
    with _reducer_cond:
        _reducer_stop = True
        _reducer_cond.notify_all()
    if _reducer_thread is not None:
        _reducer_thread.join(timeout=2.0)
    _reducer_thread = None
    with _reducer_cond:
        _pending.clear(); _dirty.clear(); _dirty_set.clear()
        _reducer_stop = False
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

> `import collections` at the top of the module is implied (`_dirty` is a `deque`).

> **Why per-connection mailboxes (not per-slot).** Two browser tabs (or an overlapping
> reconnect) on one `(node, slot)` would, with a single shared mailbox, **steal frames
> from each other** (`take` pops `_pending` to `None`). Per-connection mailboxes + a
> `publish` fan-out give every connection every latest frame. The frontend still dedups
> to one WS per `(node, slot)` *per tab*, so the common case is one mailbox.

> **Why a dedicated reducer thread (Change A).** `offer()` on the node thread is O(1)
> (snapshot meta + pointer-swap + notify); reduce+encode happen on the reducer thread, so
> a viewer never slows the node's tick or its node↔node output rate. Latest-wins on
> `_pending` bounds memory to **one live `Data` per currently-viewed slot** (plus at most
> one in-flight in the reducer), and `evict_slot` (called on the last-viewer 1→0
> transition) guarantees nothing is pinned after the viewer leaves. See §6.6 for the
> cross-thread reference-handoff lifetime + safety argument.

> **Reducer-loop correctness (must-fix).** The loop is a **predicate-loop**
> (`while not _dirty and not _reducer_stop: cond.wait()`), not a single `if … wait`, so
> there is no missed-wakeup: `offer` does `_pending[skey]=…; _dirty.append; notify()` all
> under `_reducer_cond`, and the reducer pops **both** `_dirty` and `_pending` under the
> same lock — claim and offer never interleave a half-update. A re-offer that arrives
> while the reducer is mid-reduce (outside the lock) re-adds `skey` to `_dirty`, so the
> newer frame is reduced on the next pass (latest-wins drops anything between). A
> `deque`-based round-robin (not `set.pop`) prevents one heavy slot from starving the
> previews of another in the same process.

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
            last = False
            with out_slot.viewer_lock:
                out_slot.viewer_count = max(0, out_slot.viewer_count - 1)
                last = out_slot.viewer_count == 0
            with _mailboxes_lock:
                lst = _mailboxes.get(skey)
                if lst and mb in lst:
                    lst.remove(mb)
                    if not lst:
                        _mailboxes.pop(skey, None)
            if last:
                evict_slot(node_id, slot)    # release pinned live Data + folded spec
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

> **`evict_slot` on the 1→0 transition** is the leak fix (must-fix): without it, an
> `offer()` that races a final connection close would leave a full live `Data` pinned in
> `_pending` forever (nothing pops it once `viewer_count==0` and the loop stops offering).
> The eviction runs under the reducer lock; a frame already in flight in the reducer is
> unaffected (it was already popped). `unregister_node` does the same for **all** of a
> node's slots on node teardown.

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
> per slot (last-writer-wins). The `version` field is carried for the **client's** ordering
> only and is ignored by the node. This is sound because (a) the frontend folds all
> consumers into ONE per-axis spec and debounces it (150 ms, §8.2), so the server sees a
> low-rate, already-coalesced spec stream, and (b) latest-wins on the next frame
> self-corrects any transient. This deliberately drops the contradictory "monotone per
> connection compared against a per-slot store" rule from earlier drafts.

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
server **and the one reducer thread** are **process-global, keyed by `(node_id, slot)`**;
the URL embeds `node_id`, so one port serves all co-hosted nodes. Do **not** create one
server (or one reducer) per node. Unlike `node_log`, there is **no thread-attribution
dance** (`set_process_default_node` / `bind_thread_node`): the producing node is identified
explicitly by the URL path and the `publish(node_id, slot, …)` call. The single reducer
serves every co-hosted node's slots round-robin (deque), so one slow HD downscale cannot
starve another co-hosted slot's previews.

### 4.7 Three host contexts that bootstrap the server (ONE call site)

Registration is **one line in `Node.__init__`** (§7.1), under the existing `capture_logs`
gate, beside `node_log.register_node` (`node.py:173-174`). All three contexts construct the
`Node`, so all three get a server + reducer thread **without editing the entrypoints**:

- **single-node MP child** — `_run_node_process` (`node.py:1038-1062`) → constructs Node,
- **process-group host** — `NodeProcess._run` / `_spawn_local` (`node_helpers.py:544-566`)
  → constructs each Node,
- **LOCAL / `--no-multiprocessing`** — `create_local` → constructs Node in the **manager**
  process. The data server then binds a *second* ephemeral port in the manager process,
  distinct from the bridge TCPSite **and** node_log's server (three servers in one
  process). `:0` makes collisions impossible; Step 9 adds an explicit "3 servers, no
  clash" assertion. The reducer thread is also process-global in this mode (one per manager
  process).

> Do **not** add registration calls to `_run_node_process` / `_spawn_local` /
> `create_local` — that would double-register. The "three contexts" note is only about
> *verifying the bind succeeds*, not three call sites.

### 4.8 Backpressure / latest-wins

Two independent latest-wins stages, both drop-oldest, both matching iceoryx2
`latest_wins=True`:

1. **Node → reducer** (`_pending`): `offer` overwrites the per-slot live `Data` (pointer
   swap + notify under a short lock). The node `_processing_loop` **never blocks** — it does
   not wait on the reducer. If the reducer lags (several concurrent HD downscales in one
   process), intermediate live frames are simply dropped; the heaviest slot's *preview* fps
   falls, the **processing loop is unaffected** (Change A's whole point).
2. **Reducer → browser** (per-connection `_ConnMailbox`): holds **one** encoded frame;
   `publish` overwrites (drop-oldest). A slow browser never stalls the reducer (pointer-swap
   under each mailbox's short lock) and never grows memory (one bounded immutable `bytes`
   per connection).

Memory bound: at most **one live `Data` per currently-viewed slot** in `_pending` (plus at
most one in-flight in the reducer), plus **one encoded `bytes` per live connection**. The
`evict_slot` / `unregister_node` paths (§4.2/§4.3) ensure no `Data` is pinned after the last
viewer for a slot disconnects or the node is torn down.

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

#### 5.3 Split-encode: node↔node full encode; viewer path is ONE `offer()` line (MUST)

Today `prepare_encode(data)` (`node.py:749`) and the publisher loop (`node.py:754-759`)
run **unconditionally** once a slot passes the gate. With the OR-gate, a viewer-only slot
(`subscriber_count == 0`, empty `slot.publishers`) would still run `prepare_encode` on the
full 10.6 MB `Data` every tick — **defeating the reduction goal**. Guard the full encode
on having real subscribers, and for the viewer path do nothing but **offer** the live
`Data` to the reducer (Change A — **no** reduce / encode / publish / try-except in
`node.py`):

```python
data = Data(slot.dtype, value[0], value[1])          # node.py:737, unchanged

# Node↔node iceoryx2 fan-out: ONLY when real node consumers exist. UNCHANGED behavior,
# now explicitly gated so viewer-only slots never pay the full-Data encode. This is the
# SYNCHRONOUS path — full Data is encoded into SHM on THIS thread, independent of the
# viewer path; a reducer fault/tear/latency can never affect these bytes.
if slot.subscriber_count > 0:
    size, meta_bytes = prepare_encode(data)          # node.py:749
    for pub, notif in zip(slot.publishers, slot.notifiers):
        loan = pub.loan(size)
        encode_data_into(data, loan.buffer, meta_bytes=meta_bytes)
        loan.send(); notif.notify()                  # node.py:754-759, unchanged

# Browser reduced fan-out: ONLY when a viewer is attached. ONE LINE — hand the live Data
# to the reducer thread (Change A). O(1) pointer-swap + meta snapshot; NO reduce/encode
# here, NO try/except (offer cannot meaningfully fail; reduce/encode errors are swallowed
# on the REDUCER thread, §4.2). Never marks the node errored or interrupts node↔node.
if slot.viewer_count > 0:
    node_data.offer(self.node_id, slot_name, data)
```

> Keep the existing `try/except` structure around `Data(...)` (`node.py:736-741`) intact.
> The node↔node block plus the single `offer()` line replace the old unconditional
> encode+publish block. There is **no** browser-path `try/except` in `node.py` anymore (it
> moved to the reducer thread). The `tick_error`/`_clear_error_if_healthy` accounting
> (`node.py:721, 764-767`) is driven **only** by the node↔node path and the `Data(...)`
> construction, never by the browser branch.

### 5.4 Race-safe bookkeeping (the exact guards)

- **`subscriber_count`** — single writer (messaging loop, `node.py:512-521`). **Unchanged.**
- **`viewer_count`** — written **only** by `node_data` handler threads, always under
  `slot.viewer_lock`: `+1` on WS upgrade, `max(0, -1)` in a **`finally`** around the
  handler (so abrupt `BrokenPipe`/`ConnectionReset` still decrements — a leaked count keeps
  a slot producing forever). On the `1 → 0` transition the handler also calls
  `node_data.evict_slot(node, slot)` to release any pinned live `Data` (§4.3).
- The **processing loop only READS** both counters in the gate. An `int` read is atomic
  under the GIL (a stale read costs/saves at most one tick — one extra `offer`, dropped
  harmlessly by the reducer's latest-wins) — no lock on the read side.
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

Reduction now runs on the **reducer thread** (Change A) and is **per-axis** (Change C):
the ViewSpec lists axes to reduce, each with a `max` and a `method`; `reduce_for_view`
composes them in sequence. Each reduced axis carries reconstruction info in
`meta['reduced'][str(axis)]` so the **meta inspector** can display the true original meta
off the same single per-slot stream (Change C2).

### 6.1 ViewSpec (Python) — PER-AXIS + constructor

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from goofi.data import Data, DataType

_METHODS = ("envelope", "subsample", "area")
_RICHNESS = {"envelope": 3, "area": 2, "subsample": 1}   # richest wins on fold conflict
_ORIG_COORD_CAP = 4096    # carry orig_coord verbatim only for subsample axes ≤ this

@dataclass(frozen=True)
class AxisSpec:
    axis: int                 # may be negative; canonicalized in reduce_for_view
    max: int                  # target entries on this axis (>=1)
    method: str               # 'envelope' | 'subsample' | 'area'

@dataclass(frozen=True)
class ViewSpec:
    axes: Tuple[AxisSpec, ...] = ()    # axes to reduce; unlisted axes untouched
    version: int = 0                   # client ordering only; node ignores

def viewspec_from_dict(d: dict) -> ViewSpec:
    axes = []
    for a in (d.get("axes") or []):
        if not isinstance(a, dict):
            continue
        try:
            ax = int(a.get("axis"))
            mx = max(1, int(a.get("max")))
        except Exception:
            continue
        method = a.get("method")
        if method not in _METHODS:
            continue
        axes.append(AxisSpec(axis=ax, max=mx, method=method))
    try:
        ver = int(d.get("version", 0) or 0)
    except Exception:
        ver = 0
    return ViewSpec(axes=tuple(axes), version=ver)
```

### 6.2 Entry point — compose per-axis reductions

```python
def reduce_for_view(data: Data, spec: Optional[ViewSpec]) -> Data:
    """Return a (possibly) smaller Data for `spec` by composing PER-AXIS reductions.

    INVARIANTS:
      * FAIL-OPEN: any guard trip or exception returns `data` UNREDUCED.
      * NEVER mutates `data` (node↔node publishers still encode the full object;
        and the offer-snapshot meta is already a defensive copy, §6.6).
      * Every produced array is a FRESH contiguous copy (np.ascontiguousarray);
        passthrough returns `data` itself.
      * Runs on the REDUCER thread; the lifetime/safety argument is §6.6 (cross-
        thread reference handoff), NOT "synchronous within the tick".
      * Co-reduces meta['channels']['dimD'] for EVERY reduced axis D with the SAME
        transform, so Data.__post_init__ (data.py:104) does not assert.
      * Records meta['reduced'][str(D)] = {orig_len, method, orig_coord?} per reduced
        axis D so the meta inspector can reconstruct the TRUE original meta (Change C2).
    """
    if spec is None or not spec.axes:
        return data
    try:
        if data.dtype != DataType.ARRAY:        # STRING/TABLE/scalar → passthrough
            return data
        arr = data.data
        if not hasattr(arr, "ndim") or arr.ndim == 0:
            return data
        ndim = arr.ndim

        # Canonicalize axes to positive indices; de-dup (last spec for an axis wins).
        # All three methods PRESERVE ndim, so composing in any order leaves later axis
        # indices valid — we canonicalize once up front and key meta by the positive int.
        by_axis = {}
        for a in spec.axes:
            cax = a.axis % ndim
            by_axis[cax] = a            # de-dup: last wins
        if not by_axis:
            return data

        # shallow-copy meta; deep-copy channels sub-dict before edits
        new_meta = dict(data.meta)
        ch_src = new_meta.get("channels") or {}
        new_meta["channels"] = {k: list(v) if isinstance(v, (list, tuple)) else v
                                for k, v in ch_src.items()}
        reduced_info = {}

        out = arr
        # Apply in DESCENDING canonical-axis order for determinism (ndim-preserving, so
        # order does not change indices; descending is simply a stable convention).
        for cax in sorted(by_axis.keys(), reverse=True):
            a = by_axis[cax]
            out, info = _apply_axis(out, cax, a, new_meta)
            reduced_info[str(cax)] = info

        if reduced_info:
            new_meta["reduced"] = reduced_info

        out = np.ascontiguousarray(out)
        reduced = Data(data.dtype, out, new_meta)   # constructor is the final net
        return reduced
    except Exception:
        return data                              # FAIL-OPEN
```

```python
def _apply_axis(arr: np.ndarray, axis: int, a: "AxisSpec", meta: dict):
    """Apply one axis reduction; co-reduce that axis's coord; return (out, info).
    info = {orig_len, method, orig_coord?} for the meta inspector."""
    orig_len = int(arr.shape[axis])
    ch = meta.get("channels") or {}
    coord = ch.get(f"dim{axis}")
    coord = list(coord) if isinstance(coord, (list, tuple)) else None

    if a.method == "envelope":
        out, centers = _envelope(arr, axis, a.max)
        # co-reduce coord: each bin contributes TWO body entries (min,max) ⇒ repeat 2×.
        new_coord = (list(np.repeat(np.asarray(coord)[centers], 2))
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        # Inspector reconstructs envelope axes by RANGE (orig_coord would be huge); never
        # carry orig_coord for envelope.
        return out, {"orig_len": orig_len, "method": "envelope"}

    if a.method == "subsample":
        idx = _subsample_idx(orig_len, a.max)
        out = np.ascontiguousarray(np.take(arr, idx, axis=axis))
        new_coord = ([coord[i] for i in idx]
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        info = {"orig_len": orig_len, "method": "subsample"}
        # Carry orig_coord verbatim ONLY for small subsample axes (channels/trajectory):
        if coord is not None and len(coord) == orig_len and orig_len <= _ORIG_COORD_CAP:
            info["orig_coord"] = list(coord)
        return out, info

    if a.method == "area":
        out, centers = _area_axis(arr, axis, a.max)
        new_coord = ([coord[i] for i in centers]
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        # area axes are the long pixel axes; reconstruct by RANGE, no orig_coord.
        return out, {"orig_len": orig_len, "method": "area"}

    return arr, {"orig_len": orig_len, "method": "subsample"}   # unreachable; safe default
```

### 6.3 Reduction-policy table (per-method) + per-kind axis declarations

Reductions are now **per-method**, composed per the ViewSpec's `axes`. The frontend
declares the axes per **viewer kind** (§8.2); the node applies whatever it receives.

| `method` | semantics | out length on axis | coord co-reduction | `meta['reduced'][axis]` |
|---|---|---|---|---|
| `envelope` | split axis `N` into `W=min(max,N)` bins; emit per-bin `min,max` **interleaved** → `2*W`. Preserves extremes (audio transients/clipping). Skip (return unchanged) if `N < 2*W` (ratio < 2×). **Never stride.** | `2*W` | `np.repeat(coord[bin_centers], 2)` (len `2*W`) | `{orig_len:N, method:'envelope'}` (no orig_coord) |
| `subsample` | pick `min(max,N)` unique-preserving `linspace(0,N-1)` indices; `np.take`. For channels, trajectory points. | `min(max,N)` | gather same indices: `[coord[i] for i in idx]` | `{orig_len:N, method:'subsample', orig_coord?}` (orig_coord only if `N ≤ 4096`) |
| `area` | block-mean to `min(max,N)` bins via `np.add.reduceat` over integer edges (handles non-divisor ratios). For image axes. Image threshold is **1×** (always downscale if target < source on that axis). | `min(max,N)` | gather bin centers: `[coord[c] for c in centers]` | `{orig_len:N, method:'area'}` (no orig_coord) |

**Per-kind axis lists** (mirrors frontend `capacity.ts`, §8.2 — included here so the node
side knows exactly what it will receive):

| viewer kind | input shape | declared axes |
|---|---|---|
| line 1-D | `(N,)` | `[{axis:-1, max:Wpx, method:'envelope'}]` |
| line 2-D `(C,N)` | `(C,N)` | `[{axis:0, max:rows, method:'subsample'}, {axis:-1, max:Wpx, method:'envelope'}]` |
| image | `(H,W,3|4)` | `[{axis:0, max:Hpx, method:'area'}, {axis:1, max:Wpx, method:'area'}]` |
| trajectory | `(N,2)` | `[{axis:0, max:N, method:'subsample'}]` |
| topomap / string / table / scalar | any | `[]` (no reduction — already tiny / non-array) |

> **ndim-preserving composition.** `envelope`, `subsample`, and `area` all preserve `ndim`
> (they change one axis length, never add/remove an axis). Therefore composing several axis
> reductions in sequence is well-defined regardless of order, and canonical positive axis
> indices stay valid across steps. `reduce_for_view` canonicalizes negative axes once up
> front and keys `meta['reduced']` by the canonical positive string (must-fix: a `-1` from
> one consumer and a `+1` from another that fold onto the same physical axis become the
> same key).

> **Skip thresholds.** Line/envelope skip-if-ratio<2× avoids pointless copies of a
> near-target buffer. Image/area uses a **1× threshold** so the headline HD case
> (1920×1080 → 1280×720, only 1.5×/axis) **always downscales**.

### 6.4 Meta co-reduction (the exact rule) + safety net

`Data.__post_init__` (`data.py:101-107`) asserts, **for every axis `d` that has a coord
list**, `len(meta['channels']['dimD']) == data.shape[d]`. Therefore each reduced axis with
a coord list must be co-reduced with the **same** transform (done per-axis in `_apply_axis`,
§6.2):

- **envelope** along axis `a` → reduced body length is `2*W`; co-reduced coord =
  `list(np.repeat(np.asarray(coord)[bin_centers], 2))`. Envelope **guarantees** this
  co-reduced coord so the §8.7 band keeps its x-axis (do not rely on the backstop).
- **subsample** → gather the **same indices** used on the data: `[coord[i] for i in idx]`.
- **area** → gather bin centers: `[coord[c] for c in centers]`.
- A 2-D `(C,N)` line that lists both `axis:0` (subsample) and `axis:-1` (envelope)
  co-reduces **both** dims independently; each must satisfy `len == reduced.shape[d]`.

**Backstop** (last line of defense, not the primary mechanism): immediately before
constructing the reduced `Data`, drop any `meta['channels']['dimD']` whose length ≠
`reduced.shape[d]`. The `Data(...)` constructor is the final net; any residual mismatch
raises inside `reduce_for_view`'s `try`, which **fails open** to the unreduced `data`.

```python
def _set_coord(meta: dict, dim: int, new_coord: Optional[list]) -> None:
    ch = meta.setdefault("channels", {})
    k = f"dim{dim}"
    if new_coord is None:
        ch.pop(k, None)                 # axis lost its coord → drop (body still valid)
    else:
        ch[k] = list(new_coord)
```

### 6.5 Helper bodies (numpy-only, full)

```python
def _subsample_idx(n: int, m: int) -> np.ndarray:
    """Unique-preserving linspace indices; len = min(n, m)."""
    m = min(max(1, m), n)
    idx = np.linspace(0, n - 1, m).round().astype(int)
    # unique-preserving (linspace.round can collide for tiny n); keep order, drop dups
    _, keep = np.unique(idx, return_index=True)
    return idx[np.sort(keep)]

def _envelope(x: np.ndarray, axis: int, w: int):
    """Min/max envelope along `axis`: returns (env[..,2*w,..], bin_centers[w] int idx).
    Skip (no-op) handled by the 2× ratio guard before calling."""
    n = x.shape[axis]
    w = min(max(1, w), n)
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

def _area_axis(x: np.ndarray, axis: int, m: int):
    """Block-mean (area) downscale along ONE axis to min(m, n) bins (numpy only).
    Per-axis block-mean is separable and numerically identical to a true 2-D block
    mean for non-divisor ratios (uniform per-block weights over full rectangles).
    Returns (out, bin_centers[m] int idx) for coord co-reduction."""
    n = x.shape[axis]
    m = min(max(1, m), n)
    edges = np.linspace(0, n, m + 1).astype(int)
    counts = np.maximum(1, np.diff(edges))
    f = x.astype(np.float32)
    summed = np.add.reduceat(f, edges[:-1], axis=axis)
    # broadcast the per-bin counts along `axis`
    shape = [1] * x.ndim
    shape[axis] = m
    out = summed / counts.reshape(shape)
    centers = ((edges[:-1] + np.maximum(edges[:-1] + 1, edges[1:]) - 1) // 2).astype(int)
    return np.ascontiguousarray(out.astype(x.dtype)), centers

def _area_downscale_2d(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Convenience: compose _area_axis on axis 0 (H) then axis 1 (W). Used when a
    single image ViewSpec lists both axes; identical to a 2-D block mean."""
    out, _ = _area_axis(img, 0, out_h)
    out, _ = _area_axis(out, 1, out_w)
    return np.ascontiguousarray(out)
```

> `reduce_for_view` composes `_apply_axis` per declared axis; for the image kind this is
> `_area_axis` on axis 0 then axis 1, which equals `_area_downscale_2d` and equals a true
> 2-D block mean (separability verified numerically: max abs diff ~3e-8 for non-divisor
> ratios like 7→3, 5→2). `reduceat` with `linspace` integer edges produces no empty/
> duplicate bins as long as `out_dim ≤ source_dim` (enforced by `min()`).

### 6.6 Lifetime + safety (the load-bearing guarantee — private-snapshot handoff)

> **Replaces the original "synchronous within the tick" guarantee.** Change A moves
> reduce+encode to the reducer thread. To make that race-free for EVERY node — including
> ones that mutate persistent state in place — `offer()` hands the reducer a **private
> snapshot taken on the node thread**, never a reference to live node state.

`offer(node, slot, data)` runs on the node's `_processing_loop` thread and builds a fresh
`Data` that aliases nothing the node retains: a **contiguous copy of the array** plus a
**copied meta** (shallow + the `channels` sub-dict). Only that snapshot is stored in
`_pending` (latest-wins) and read by the reducer. The safety argument is then total and
trivial:

1. **No cross-thread aliasing.** The reducer only ever touches the private snapshot. The
   node may rebind or mutate its own buffers on any later tick with zero effect on
   in-flight reductions.

2. **No torn reads — ever.** The snapshot's array memcpy executes on the node thread,
   *serialized with that node's own mutations* (same thread, different ticks): the node is
   never writing the source while the copy reads it. This is exactly what makes it correct
   for in-place mutators like **`LatentRotator`** (`nodes/misc/latentrotator.py:89-95`,
   which `+=`/`/=` `self.cumulative_vector` and returns that same object — and
   `Data._configure_array` does NOT copy `ndim>=1` arrays, `data.py:88-89`) and meta-reusers
   like **`pca.py`** (returns `self.meta`; a *reference* handoff would race the reducer's
   dict copy → `RuntimeError: dictionary changed size during iteration`). Snapshotting on
   the node thread closes both at the source.

3. **No use-after-free, no node-state or node↔node corruption.** The snapshot owns its
   memory. The node↔node path encodes the *original* `Data` into SHM synchronously on the
   node thread (§5.3), wholly independent of `offer()`.

```python
def _snapshot_for_offer(data: "Data") -> "Data":
    """Run on the NODE thread. Return a Data that aliases nothing the node retains:
    a contiguous COPY of the array + a copied meta (shallow + channels sub-dict).
    Always copies — one array memcpy on the node thread, far cheaper than the
    reduce+encode it lets the reducer thread run OFF the tick."""
    from goofi.data import Data
    arr = data.data
    meta = dict(data.meta)
    ch = meta.get("channels")
    if isinstance(ch, dict):
        meta["channels"] = {k: list(v) if isinstance(v, (list, tuple)) else v
                            for k, v in ch.items()}
    body = np.ascontiguousarray(arr).copy() if hasattr(arr, "ndim") else arr
    return Data(data.dtype, body, meta)
```

**Cost.** One array memcpy per offered frame on the node thread: ~1–2 ms for a 6 MB HD
frame against a >=33 ms tick (~3–6%), far less for EEG/PSD/audio. The *expensive* work — the
reduction (`envelope` / `area` `reduceat`) and the encode — runs on the reducer thread, so
the tick is never slowed by them (Change A's goal). The snapshot also captures the **full
original meta** for the metadata inspector (§6.4 / §8): the reducer derives `meta['reduced']`
from this pristine copy, so the inspector reconstructs the true meta regardless of any later
node mutation.

> **Deliberately always-copy (not threshold / zero-copy).** An earlier draft copied only
> arrays below 8 MB and left large (HD) arrays as zero-copy references, assuming "every large
> producer allocates a fresh frame per tick." That is precisely the kind of unaudited
> whole-tree invariant whose small-array version was already FALSE (`LatentRotator`), so the
> spec ships **always-copy** for correctness-by-default. Skipping the copy for large frames is
> a valid *future* optimization — but only behind a profile showing the memcpy is a real
> bottleneck AND a maintained contract that large-array producers never mutate in place; it is
> not a baseline assumption.

**Backpressure / coupling (changed from the original).** Reduction is **decoupled** from
the tick: `offer()` is O(1) (snapshot + pointer-swap + notify), so viewing a slot does
**not** slow that node's tick or its node↔node output rate (this is Change A's stated goal,
a genuine improvement over the old inline §5.3/§6.6 design). If the reducer lags, latest-wins
drops intermediate frames → lower preview fps on the heaviest slot only; the processing loop
is unaffected.

> **Tests (replace the old `np.shares_memory` test).**
> - A node that mutates its **returned array's contents** in place after return (a
>   `LatentRotator`-shaped fixture, not a rebinding one) cannot crash the reducer and cannot
>   corrupt or stall node↔node delivery; the reduced output is a **fresh** array.
> - Node↔node integrity is independent of the reducer: with the reducer made to raise on
>   every frame, a node↔node subscriber still receives full, correct frames (node↔node bytes
>   are encoded synchronously on the node thread, §5.3).
> - `offer` twice rapidly for one slot ⇒ exactly the **latest** is reduced; no `skey` left
>   orphaned in `_dirty` with a `None` `_pending`.
> - View a slot, disconnect ⇒ `_pending` has no entry for that slot and no full `Data` is
>   pinned (`evict_slot`).
> - After `_reset_for_tests` there is no live thread named `goofi-data-reducer` and no live
>   `goofi-data-ws` server thread.

---

## 7. Negotiation Protocol

### 7.1 Endpoint advertising via `STATE_UPDATE` / snapshot (PORT, not URL — Change B)

- **Mint** (`node.py:__init__`, ONE line beside `node.py:173-174`):
  ```python
  self._data_port: Optional[int] = None        # init near node.py:150 beside _log_port
  if capture_logs:                              # same non-headless gate as logs
      self._log_port  = node_log.register_node(node_id)        # returns int port now
      self._data_port = node_data.register_node(node_id, self) # returns int port
  ```
  (`node_log.register_node` is **edited** to return a port int as well — Change B; the old
  full-URL return is removed everywhere, see §9.)
- **Advertise** (`node.py:_push_state`, beside `node.py:423`):
  ```python
  state = { ..., "log_port": self._log_port,
                 "data_port": self._data_port }
  ```
  The ports are **static**, so they ride the **first** post-setup `_push_state` (which
  always fires because `setup()` marks the node dirty). Invariant: *a node always advertises
  `log_port` and `data_port` on its first post-setup state push* — identical dependency to
  the old `log_endpoint`, which already works.
- **Relay** (`bridge/control.py:on_state`, beside `control.py:343`):
  ```python
  "payload": { ..., "log_port":  message.content.get("log_port"),
                    "data_port": message.content.get("data_port") }
  ```
- **Snapshot** (`bridge/schemas.py:describe_node_instance`, beside `schemas.py:99`):
  ```python
  "log_port":  (ref.serialized_state or {}).get("log_port"),
  "data_port": (ref.serialized_state or {}).get("data_port"),
  ```

> One **per-process port** (not a per-slot map): the port is per-process, so all slots
> share it; the browser composes `ws://${location.hostname}:${data_port}/<node>/<slot>`.
> This keeps `data_port` plumbing byte-identical to `log_port` (a single scalar `!==
> undefined` merge) and survives dynamic `output_slots` re-declaration without churn.

> **Atomic port-vs-URL switch (must-fix).** The change from a full-URL `log_endpoint`
> string to an int `log_port` (and the new `data_port`) touches six coordinated sites:
> `node.py` mint + `_push_state`; `bridge/control.py` relay; `bridge/schemas.py` snapshot;
> `frontend/src/lib/api/control.ts` types + `state_update` payload; `graph.svelte.ts` merge;
> `logStream.svelte.ts` URL composition. Land them in **one commit** and run
> `git grep -n log_endpoint` and `git grep -n register_node` across the Python + TS + test
> tree first — any missed site silently dark-connects (`new EventSource(undefined)` /
> compose-against-undefined). The old `log_endpoint` field is fully **removed**, not left
> dangling. Add an assertion that `node.log_port` / `node.data_port` are ints and that
> `logStream` composes a valid `http`/`ws` URL.

### 7.2 The data WS messages (per-axis ViewSpec — Change C)

| dir | when | payload |
|---|---|---|
| C→S | connect URL query (zero-RTT seed) | `<base>/<node>/<slot>?spec=<base64url(JSON ViewSpec)>` where ViewSpec = `{axes:[{axis,max,method}], version}` (≤ `_SEED_MAX=4096` chars; oversize ⇒ omit, rely on the post-connect message) |
| C→S | WS **TEXT** on every (re)connect and renegotiation | `{ "op":"view", "spec": {axes,version}, "v": <int> }` |
| S→C | WS **BINARY** | a GOOF-encoded **reduced** `Data` frame (`$lib/codec/decode` unchanged) |
| C↔S | keepalive / liveness | WS `ping`/`pong` |

- **First-frame seed**: the `?spec=` query sets `_specs[(node,slot)]` **before** the first
  tick, so the very first reduced frame is correctly sized (critical — one unreduced
  44.1 kHz frame is 10.6 MB). Base64url alphabet, `=` padding restored on decode (§4.4).
  Example seed payload: `{"axes":[{"axis":-1,"max":1600,"method":"envelope"}],"version":3}`.
- **Versioning**: `version`/`v` is **client-side ordering only**; the **server ignores it**
  and applies last-received-wins.
- **Reconnect**: no server resume. The client re-resolves `data_port` from graph state,
  **re-composes the URL from `location.hostname`**, re-opens, re-sends the seed (query)
  **and** an immediate `{op:"view"}`. The server resets `viewer_count` correctly across
  reconnect (decrement-in-`finally` + `evict_slot` on the 1→0, increment on new upgrade),
  and the new connection gets its **own** mailbox.

---

## 8. Frontend Changes

### 8.0 Ownership split (RESOLVED)

- **`dataStream.svelte.ts` OWNS the per-`(node,slot)` WebSocket lifecycle** (open / close /
  move), exactly as `logStream.svelte.ts` owns its EventSources. It resolves `data_port`
  from graph state, **composes the URL from `location.hostname`** (Change B), and reconciles
  sockets against the active need set.
- **`data.ts` is repurposed to a stateless connection helper** invoked *by* `dataStream`:
  it builds the URL from a passed-in resolver (host + `data_port` + slot + seed spec), opens
  the socket, decodes frames, exposes an `updateSpec(spec)` method to push `{op:"view"}` on
  an already-open socket, and fans decoded frames to listeners. It no longer reads
  `location.host` directly for the *whole* URL — it takes a `() => string|null` endpoint
  resolver from `dataStream` (which composed it from `location.hostname` + port). It no
  longer owns refcount/dedup of the *need* set (that moves to `dataStream`), but **keeps
  per-socket listener fan-out** so multiple `ViewerFeed`s on one `(node,slot)` share one
  socket.

This removes the "two owners for one socket" contradiction: **`dataStream` owns sockets;
`data.ts` is its transport primitive.**

> **Dev-mode / proxy note (Change B, must-fix).** Only **`/control`** traverses the Vite
> dev proxy (it uses `location.host` → 5173 → bridge:8000). **`/data` and the log SSE go
> DIRECT** to the node ports (`ws://${location.hostname}:${data_port}` /
> `http://${location.hostname}:${log_port}`), bypassing the proxy — this is intentional and
> matches how logs already worked. With the new `GOOFI_BIND_HOST=0.0.0.0` default the node
> listens on all interfaces, so `localhost:<nodeport>` and `<LANIP>:<nodeport>` both reach
> it; the developer must open the SPA on a hostname that also reaches the node interface
> (with the 0.0.0.0 default this holds for `localhost` and LAN IP). Do **not** "fix" control
> to be direct — it stays same-origin-proxied so HTTPS tunnels keep working for the control
> plane.

### 8.1 New: `frontend/src/lib/stores/dataStream.svelte.ts`

Mirror `logStream.svelte.ts`: a process-wide singleton with a single `$effect.root`
`reconcile()` that:

1. reads `graph().nodes` → `ports = Map<nodeName, data_port|null>`,
2. reads the active need set `(node, slot)` (populated by `ViewerFeed` via `setNeed`/
   `release`, like logStream's `setNeeds`/`release`, mutations wrapped in `untrack`),
3. for each need whose `data_port` is **known**, composes the URL from `location.hostname`
   and ensures an open connection (`data.ts.connect(...)`),
4. closes any connection no longer needed **or** whose composed endpoint
   `!== conn.endpoint` (endpoint **moved** — node restarted on a new port), mirroring
   `logStream.svelte.ts:84`,
5. subscribes to `thalamus.viewSpecFor(node, slot)` changes and calls `conn.updateSpec(spec)`
   to push `{op:"view"}` on the live socket.

URL composition (single helper, shared shape with `logStream`):

```ts
function composeDataUrl(host: string, port: number, node: string, slot: string): string {
  const h = (typeof location !== 'undefined' && location.hostname) || '127.0.0.1'; // file:// guard
  const scheme = (typeof location !== 'undefined' && location.protocol === 'https:') ? 'wss' : 'ws';
  return `${scheme}://${h}:${port}/${encodeURIComponent(node)}/${encodeURIComponent(slot)}`;
}
```

Defers opening until `data_port` is known (null until first `STATE_UPDATE`).
Reconnect/backoff lives in `data.ts` but **re-resolves the endpoint from `dataStream` on
each retry** (never a cached/stale URL) — `data.ts.connect` is given a `() => endpoint`
resolver, not a frozen string. `dataStream` exposes a `(node,slot)`-keyed decoded-frame
source that `frames.ts` consumes (so `frames.ts` never sees the endpoint).

> `location.hostname` (not `location.host`) is correct because the node port differs from
> the page port. The `file://` case (`location.hostname === ''`) falls back to `127.0.0.1`.

### 8.2 New: `frontend/src/lib/stores/thalamus.svelte.ts` (PER-AXIS — Change C)

```ts
export type ReduceMethod = 'envelope' | 'subsample' | 'area';
export interface AxisSpec { axis: number; max: number; method: ReduceMethod; }
export interface ViewSpec { axes: AxisSpec[]; version: number; }

// addConsumer(node, slot, id: symbol, spec: ViewSpec) / updateConsumer / removeConsumer
// viewSpecFor(node, slot): ViewSpec   // folds all live consumers → ONE per-axis spec
```

**Per-kind axis declarations (capacity.ts).** Each viewer declares its axes + per-axis max
from its kind and canvas size:

```
line 1-D:        [{ axis: -1, max: Wpx, method: 'envelope' }]
line 2-D (C,N):  [{ axis: 0,  max: rows, method: 'subsample' },
                  { axis: -1, max: Wpx,  method: 'envelope'  }]
image:           [{ axis: 0,  max: Hpx, method: 'area' },
                  { axis: 1,  max: Wpx, method: 'area' }]
trajectory:      [{ axis: 0,  max: N,   method: 'subsample' }]
topomap/string/table/scalar: []   // no reduction
```

Capacity math: `Wpx = max(64, ceil(canvasCssPx * devicePixelRatio))`; image
`Hpx/Wpx = max(1, h*dpr)/max(1, w*dpr)`; `rows = visible channel rows`. At 0 px, clamp to
the 64 floor and **do not register a need** until the element has area.

**Folding = PER-AXIS** (replaces the old single-mode + mode-precedence rule):

- Group all live consumers' `axes` by **canonical** axis (resolve negatives against the
  known shape if available; otherwise key by the raw int and let the node canonicalize —
  the node re-canonicalizes against actual `ndim`, §6.2).
- For each axis, `max = max(consumer.max)` over consumers that list that axis.
- On a **method conflict** for an axis, pick the **richest**: `envelope > area > subsample`
  (richest preserves the most information; a richer representation is a superset a poorer
  viewer can still render). The old global "mode precedence image>line>…" rule is **gone**.
- Bump `version` on each effective change. **Debounce 150 ms** + **hysteresis**: round
  width up to 64 px; renegotiate only on **> 25 % grow** or **> 50 % shrink**.

### 8.3 Edited: `frontend/src/lib/api/data.ts`

- New shape: `connect(resolveEndpoint: () => string|null, getSpec: () => ViewSpec)`
  returns `{ endpoint: string, onFrame(cb), updateSpec(spec), close() }`.
  (`resolveEndpoint` returns the fully-composed `ws://host:port/node/slot` base from
  `dataStream`, §8.1.)
- URL: `${endpoint}?spec=${b64urlSpec}` (seed from `getSpec()`). Build base64url from
  `JSON.stringify(spec)` where `spec = {axes,version}`; if the encoded query exceeds 4096
  chars, **omit** `?spec=` and rely on the post-open `{op:"view"}`.
- On `open`: send `{op:"view", spec: getSpec(), v}` immediately; on every `updateSpec`,
  send again with an incremented `v`.
- On reconnect (existing 250 ms→5000 ms backoff): call `resolveEndpoint()` again; if it
  returns a different/null endpoint, stop and let `dataStream` re-drive. Re-send seed +
  `{op:"view"}` on the new socket.
- `binaryType = 'arraybuffer'`; `decodeData(e.data)` **unchanged** (reduced frames are the
  same GOOF format, just smaller).

### 8.4 Edited: `frontend/src/lib/api/control.ts`

Replace the old `log_endpoint?: string | null` with `log_port?: number | null` and add
`data_port?: number | null` to `NodeInstanceInfo` (~`:38`) and to the `state_update`
`ControlEvent` payload (~`:103`). Remove every `log_endpoint` reference (must-fix: grep the
TS tree).

### 8.5 Edited: `frontend/src/lib/stores/graph.svelte.ts`

In the `state_update` handler (beside the old `log_endpoint` merge, ~`:191`), replace the
endpoint merge with port merges:
```ts
if (ev.payload.log_port  !== undefined) t.log_port  = ev.payload.log_port;
if (ev.payload.data_port !== undefined) t.data_port = ev.payload.data_port;
```
Add `log_port?: number | null` and `data_port?: number | null` to the node type (remove
`log_endpoint`). `hello` / `node_added` already carry both via `NodeInstanceInfo`.

### 8.6 Edited: `frontend/src/lib/viewers/ViewerFeed.svelte`

At the `visible` toggle (IntersectionObserver, ~`ViewerFeed.svelte:27-44`): mint
`const consumerId = Symbol()`, observe the canvas via `ResizeObserver`, and
`thalamus.addConsumer/updateConsumer/removeConsumer` (with the per-axis ViewSpec from
capacity.ts, §8.2) on the `visible && expanded` effect. Register/release the `(node, slot)`
**need** with `dataStream` on the same effect. `frames.ts` (RAF coalesce to 1 frame/paint)
and `subscribeFrames(node, slot, cb)` stay **unchanged in signature** — they call into
`dataStream`'s per-`(node,slot)` frame stream, so the endpoint/ViewSpec machinery is
entirely upstream of `frames.ts`.

> The move-handling lives in **`dataStream`**; `frames.ts` keeps its `(node, slot)`
> signature because `dataStream` presents a `(node, slot)`-keyed frame source. `frames.ts`
> source code is unchanged.

### 8.7 Edited: `frontend/src/lib/stores/logStream.svelte.ts` (Change B)

`logStream` previously connected via `new EventSource(node.log_endpoint)` (a backend-given
full URL). It now **composes** the URL from `location.hostname` + the advertised `log_port`:

```ts
const h = (typeof location !== 'undefined' && location.hostname) || '127.0.0.1';
const scheme = (typeof location !== 'undefined' && location.protocol === 'https:') ? 'https' : 'http';
const url = `${scheme}://${h}:${node.log_port}/${encodeURIComponent(node.name)}`;
es = new EventSource(url);
```

`reconcile` now keys "endpoint moved" off `log_port` (not `log_endpoint`). Remove the
`log_endpoint` read. This brings logs into the **same host scope** as data (both direct to
the node port on `location.hostname`).

### 8.8 ArrayViewer envelope band (per-axis — Change C)

`frontend/src/lib/codec/decode.ts` **already exposes `meta`** on `DataFrame`
(`decode.ts:40`: `meta: Record<string, unknown>`), so `meta.reduced` survives decode with
**no decode change**. Add a typed accessor in the viewer:

```ts
type ReducedAxis = { orig_len?: number; method?: string; orig_coord?: unknown[] };
const reduced = (f.meta.reduced ?? null) as Record<string, ReducedAxis> | null;
```

**Authority rule (must-fix): the viewer reads layout from the RECEIVED frame, never from
its own requested ViewSpec.** Because of per-axis fold (richest wins), a viewer can receive
a richer representation than it asked for (e.g. a `subsample` request folded to `envelope`
by another consumer, or a larger `max`). The body layout is whatever the folded spec
produced. In `ArrayViewer.svelte`, resolve the displayed sample axis to its canonical
positive index and check `reduced?.[String(axis)]?.method`:

- when `method === 'envelope'`, the body length along that axis is `2*W` interleaved
  `min,max`. **De-interleave** in the viewer: `mins = values[0::2]`, `maxs = values[1::2]`,
  and render a **min/max band** (uPlot fill between the two series) instead of a single
  line, so audio transients/clipping stay visible.
- otherwise (`subsample`/`area`/absent), render the single series as today.

> Test: two array viewers of differing `max` on one slot fold to one `envelope` buffer of
> `2*maxA`; **both** render a correct band from the single folded buffer (neither assumes
> its own requested `max`).

### 8.9 New: MetadataPanel is reduction-aware (Change C2) — shows the TRUE original meta

`frontend/src/lib/editor/MetadataPanel.svelte` consumes the **same** per-slot stream as the
viewer (`subscribeFrames(node, slot)`, L35) and renders `lastFrame.meta` (L79) and
`lastFrame.data.shape` (L72). With reduction, that stream carries **reduced** frames — so
the panel must **reconstruct and display the original meta** off the single stream (no
second full stream). Two surfaces must be reconstructed:

1. **The shape line** (`MetadataPanel.svelte:72`, currently `lastFrame.data.shape`, which
   `decode.ts:124` reads from the **reduced** body header). Compute a displayed shape from
   the reduced shape with each reduced axis `d` replaced by `meta.reduced[d].orig_len`
   (envelope: the reduced body axis is `2*W`; replace it with `orig_len`, not by halving the
   `2*W`):
   ```ts
   function reconstructShape(frame: DataFrame): number[] {
     const shape = [...frame.data.shape];
     const red = (frame.meta.reduced ?? {}) as Record<string, ReducedAxis>;
     for (const [k, info] of Object.entries(red)) {
       const ax = Number(k);
       if (Number.isInteger(ax) && ax >= 0 && ax < shape.length && info.orig_len != null) {
         shape[ax] = info.orig_len;
       }
     }
     return shape;
   }
   ```

2. **The raw meta JSON block** (`MetadataPanel.svelte:79`, which also includes
   `meta['shape']` — `Data.__post_init__` overwrites `meta['shape']` with the **reduced**
   shape at `data.py:92`). Before rendering, deep-clone `meta`, overwrite `shape` with the
   reconstructed original, restore `channels['dimD']` per reduced axis (from `orig_coord`
   when present, else synthesize an index/`linspace` range of length `orig_len`), and
   **delete** the `reduced` key so artifacts are hidden:
   ```ts
   function reconstructMeta(frame: DataFrame): Record<string, unknown> {
     const meta = structuredClone(frame.meta) as Record<string, unknown>;
     const red = (meta.reduced ?? {}) as Record<string, ReducedAxis>;
     const shape = reconstructShape(frame);
     meta.shape = shape;
     const channels = { ...((meta.channels as Record<string, unknown>) ?? {}) };
     for (const [k, info] of Object.entries(red)) {
       const ax = Number(k);
       const key = `dim${ax}`;
       if (info.orig_coord && info.orig_coord.length === info.orig_len) {
         channels[key] = info.orig_coord;                    // verbatim (small subsample axes)
       } else if (info.orig_len != null) {
         channels[key] = Array.from({ length: info.orig_len }, (_, i) => i); // index range
       }
     }
     meta.channels = channels;
     delete meta.reduced;
     return meta;
   }
   ```
   For very long reconstructed coord arrays, **truncate with an ellipsis** in the `<pre>`
   block (display only) so a long subsample/index axis does not print thousands of entries.

**Net result:** the meta inspector shows **exactly an unreduced frame's meta** — original
shape, original (or index-range) coords, no `reduced` artifact — even though it consumes the
single reduced stream. `decode.ts` requires **no change** (`meta` already survives decode as
`Record<string, unknown>`). The `orig_coord` payload is bounded: it is carried verbatim only
for **subsample** axes with `orig_len ≤ 4096` (channels/trajectory — small); envelope/area
axes (the long sample/pixel axes) carry only `orig_len` and are reconstructed by range, so
the reduced frame never re-introduces the large payload the reduction exists to remove.

---

## 9. Manager / Bridge Changes + Removals

### 9.0 Manager: export `GOOFI_BIND_HOST` before any spawn (Change B)

In `Manager.__init__` (`manager.py`, before `start_bridge` at `:158` and before
`post_init`/`load` at `:162`), **inside the existing `if not self.headless:` block**, set
the env var from the resolved bind host:

```python
if not self.headless:
    os.environ["GOOFI_BIND_HOST"] = bridge_host   # default 0.0.0.0 (manager.py:618)
    start_bridge(self, host=bridge_host, port=...)
    ...
```

- `Manager.__init__` runs entirely **before any node spawn** (`manager.py:100-129`), and
  children inherit `os.environ` on both spawn (Windows) and fork (Linux), so **every** node
  — including lazily added ones later — reads the already-set `GOOFI_BIND_HOST` at
  `register_node` time. LOCAL/`--no-multiprocessing` shares the same live `os.environ`.
- Setting it **only in non-headless mode** keeps the env from leaking `0.0.0.0` into the
  many **headless** test Managers, so a later standalone-node test still gets the
  `127.0.0.1` default. The test reset fixture also `pop`s `GOOFI_BIND_HOST` (§10 / Step 10)
  to prevent cross-test contamination.
- **Startup URL:** when composing the printed "Open <url>" message, map `0.0.0.0` →
  `127.0.0.1` (or the machine's primary LAN IP) so the instruction is navigable
  (`0.0.0.0` is not). Print a one-line LAN security banner when the bind host is `0.0.0.0`
  (see §10).

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

### 9.2 EDIT (Change B — host scope)

- **`src/goofi/manager.py`** — export `GOOFI_BIND_HOST` (§9.0); map `0.0.0.0`→`127.0.0.1`
  for the printed URL; LAN security banner.
- **`src/goofi/node_log.py`** — `_LogServer` binds
  `os.environ.get("GOOFI_BIND_HOST", "127.0.0.1")` (was hardcoded `127.0.0.1`, `:237`);
  `register_node` returns the **port int** (was a full URL string, `:289`). Remove the
  `http://127.0.0.1:<port>/<node>` composition.
- **`src/goofi/node.py`** — store `self._log_port` / `self._data_port`; advertise
  `log_port` / `data_port` in `_push_state` (replacing `log_endpoint`).
- **`src/goofi/bridge/control.py`** — relay `log_port` + `data_port` (replacing
  `log_endpoint`).
- **`src/goofi/bridge/schemas.py`** — snapshot `log_port` + `data_port`.
- **`frontend/src/lib/api/control.ts`** — `log_port` / `data_port` types (replace
  `log_endpoint`).
- **`frontend/src/lib/stores/graph.svelte.ts`** — merge `log_port` / `data_port`.
- **`frontend/src/lib/stores/logStream.svelte.ts`** — compose the SSE URL from
  `location.hostname` + `log_port` (§8.7).

### 9.3 KEEP (control plane + graph ownership)

- `/control` WS (`bridge/control.py`), RPC dispatch, `_snapshot`, node/link add/remove
  events, `_wire_node_status` STATE_UPDATE + PROCESSING_ERROR relay, and the **log SSE**
  relay (`control.py:343`, now carrying `log_port`).
- `Manager` graph + link ownership (`NodeContainer`, `_links`, `add_node`/`remove_node`/
  `add_link`/`remove_link`/`save`/`load`).
- `NodeRef.register_subscriber`/`unregister_subscriber` and their ctrl handling
  (`node.py:512-521`) — **KEEP**: used by real node↔node links and asserted by tests
  (`tests/test_node.py:136-148`, `tests/test_manager.py:74`).
- Static SPA serving + SPA fallback + `no_cache` middleware. **`/control` stays
  same-origin-proxied** in dev (Vite proxy) — do not make it direct.

### 9.4 Confirm UNTOUCHED

- **Node↔node iceoryx2 publish** (`node.py:754-759`, now guarded by `subscriber_count > 0`
  but byte-for-byte identical work when subscribers exist), `_ensure_output_endpoints`,
  `SUBSCRIBE_INPUT` wiring, `transport.py`, and **`codec.py`** — unchanged. The reduced
  browser frame is a **separate** encode of a **separate** reduced `Data` on the **reducer
  thread**; the full `Data` still flows to iceoryx2 publishers synchronously on the node
  thread (CLAUDE.md §13). `frames.ts` and `frontend/src/lib/codec/decode.ts` — unchanged.

---

## 10. Edge Cases and Risks

- **Process-group host**: one process, many nodes, **one port + one reducer thread**, keyed
  by `(node_id, slot)`. One server, never one-per-node; the single reducer serves all
  co-hosted slots round-robin (deque) so a heavy slot cannot starve another's previews.
- **LOCAL / `--no-multiprocessing`**: node + data server + log server + bridge TCPSite all
  in the manager process on distinct `:0` ports, all bound to `GOOFI_BIND_HOST`.
  Loopback-P2P, manager not transcoding. Step 9 asserts no clash.
- **Multi-viewer dedup (one tab)**: two `ViewerFeed`s of one slot → one need → one socket
  (`dataStream`) → one folded per-axis ViewSpec → reducer reduces once → fanned to the one
  connection mailbox → `data.ts` fans to both listeners.
- **Multi-tab / overlapping reconnect**: each tab/socket gets its **own** connection
  mailbox; `publish` fans the same bytes to all → no frame-stealing. `viewer_count` counts
  each connection; last close (incl. abrupt, via `finally`) returns it to 0 and triggers
  `evict_slot`.
- **Reconnect**: WS has no resume; `data.ts` backoff re-resolves the endpoint (re-composing
  from `location.hostname` + current `data_port`) and re-sends the ViewSpec. Reader thread
  is unblocked by `socket.shutdown(SHUT_RDWR)` on handler exit → no thread leak.
- **Node restart**: new ephemeral port ⇒ `data_port` changes ⇒ `dataStream` reconcile
  closes the stale socket and reopens against the new composed URL.
- **Dtype change mid-stream**: each frame self-describes via the GOOF header/meta;
  `reduce_for_view` composes per-axis on the **actual** shape (canonicalizing negative
  axes against `ndim`), fail-open on mismatch (e.g. an image arriving where a 1-D envelope
  axis is listed → the area/envelope still applies per axis or fails open). Viewer reacts on
  decode.
- **0 px layout**: gated on `visible && expanded`; thalamus clamps to a 64 floor and the
  consumer registers **no need** until it has area.
- **HiDPI**: capacity uses `devicePixelRatio`; hysteresis prevents renegotiation thrash.
- **Backpressure**: two latest-wins stages (node→reducer `_pending`, reducer→browser
  per-connection mailbox), both drop-oldest, matching iceoryx2. The reducer thread is the
  rate limiter for previews; the **processing loop is never slowed** by a viewer (Change A).
- **Host scope / LAN reachability (Change B)**: node servers bind `GOOFI_BIND_HOST` (default
  `0.0.0.0`) and send `Access-Control-Allow-Origin: *`; the frontend composes URLs from
  `location.hostname`, so a LAN browser reaches the node ports directly. This is the **same
  network scope the manager bridge already had** (`--bind` defaults to `0.0.0.0`,
  `manager.py:618`); Change B only brings **logs** up to that scope (they were previously
  `127.0.0.1`-only, narrower than the manager). **Security note:** binding `0.0.0.0` exposes
  node logs + data to the LAN with CORS `*` and **no auth** — single-user / trusted-LAN app
  (CLAUDE.md §13). The manager prints a one-line LAN banner at startup when bound to
  `0.0.0.0`. The old "127.0.0.1 only / not reachable from a remote browser" limitation is
  **removed**.
- **Mixed-content / HTTPS limitation (Change B)**: the SPA over plain `http` composes `ws`;
  over `https` composes `wss`. **However**, the node data/log servers are **plaintext stdlib
  servers on separate ephemeral ports** that an HTTPS-terminating tunnel (ngrok/Cloudflare,
  forwarding only `443`→bridge) does **not** proxy — so `wss://<tunnelhost>:<nodeport>`
  cannot connect. **Supported scope: `http` origin → `ws`/`http` node ports on localhost or
  a trusted LAN.** HTTPS/tunnel deployments support **only `/control`** (same-origin
  proxied); the data + log planes are **unsupported** behind a TLS tunnel (would require a
  relay or per-node TLS — out of scope). This limitation is documented, not silently
  shipped.
- **Risks (with mitigations)**: (1) forgetting the OR-gate ⇒ silent no-frames — Step 4
  acceptance asserts a frame arrives via `node_data.publish` (off the reducer); (2) leaked
  `viewer_count` on abrupt close ⇒ decrement-in-`finally` + a Step 9 leak test; (3) reducer
  tearing a node↔node `Data` ⇒ impossible by construction (node↔node encodes synchronously
  on the node thread, §5.3) + the Change-A contract test (a node mutating its returned
  array's contents after return cannot crash the reducer; reduced output is a fresh array);
  (4) pinned live `Data` after last viewer leaves ⇒ `evict_slot` on the 1→0 transition +
  `unregister_node` clears `_pending`/`_dirty` + a leak test; (5) reducer thread leak across
  the test suite ⇒ `_reset_for_tests` stops + joins `goofi-data-reducer` and a Step 9/10
  assertion of no live reducer/server thread after reset; (6) meta co-reduction miss ⇒
  envelope guarantees the coord, backstop drops mismatches, `Data(...)` is the net,
  fail-open; (7) hand-rolled RFC6455 (unmask, 64-bit length, single write-lock, reader-
  teardown) — isolate + unit-test at 100 B/70 KB/3 MB; (8) `_reset_for_tests` parity or the
  128-test suite flakes on leaked threads/ports; (9) Lock-as-field pickle crash ⇒ lazy
  `viewer_lock` property, never a dataclass field; (10) `GOOFI_BIND_HOST` leaking `0.0.0.0`
  into standalone-node tests ⇒ set only in non-headless mode + `pop` in the reset fixture;
  (11) missed port-vs-URL site ⇒ grep `log_endpoint`/`register_node` and land the switch in
  one commit (§7.1).

---

## 11. Ordered, Commit-Sized Implementation Plan

> Backend-first; each step independently testable. Run `pytest tests/` after each backend
> step (128 tests must stay green).

**Step 1 — `OutputSlot.viewer_count` (plain int) + lazy `viewer_lock` + OR-gate + split-encode (offer).**
*What*: add `viewer_count` field and the `viewer_lock` property to `OutputSlot`
(`node_helpers.py`, ~`:170`); change the gate at `node.py:731` to
`subscriber_count == 0 and viewer_count == 0`; split the post-`Data` block so
`prepare_encode`+publisher loop is guarded on `subscriber_count > 0`, and the viewer path is
the single line `if slot.viewer_count > 0: node_data.offer(self.node_id, slot_name, data)`
(§5.3 — no reduce/encode/try-except in node.py).
*Files*: `src/goofi/node_helpers.py`, `src/goofi/node.py`.
*Acceptance*: `pytest tests/` green; `pickle.dumps(OutputSlot(DataType.ARRAY))` and a
`copy.deepcopy` succeed (no Lock in pickle); a unit test sets `slot.viewer_count = 1` on a
slot with `subscriber_count == 0`, monkeypatches `node_data.offer`, ticks the loop, and
asserts `node_data.offer` is called (NOT that `process()` re-encoded).

**Step 2 — `node_data.py` server + reducer subsystem + framing + `_reset_for_tests`.**
*What*: new module per §4: per-process `ThreadingHTTPServer` bound to
`GOOFI_BIND_HOST` (Change B), hand-rolled RFC6455 (handshake, client unmask, server
7/16/64-bit write via `_send_frame`, single per-conn `_write_lock`, ping/pong/close, reader
thread + `shutdown` teardown), per-connection `_ConnMailbox` + `(node,slot)->[mailboxes]`
registry, `_specs` last-received-wins, **the reducer subsystem** (`_pending`, `_dirty`
deque + `_dirty_set`, `_reducer_cond`, `offer`, `_reducer_loop`, one `goofi-data-reducer`
daemon thread started in `_ensure_server_locked`, `_snapshot_for_offer`, `evict_slot`),
`register_node(node_id, node) -> int`/`unregister_node`/`viewspec_for`/`publish`,
`_reset_for_tests` (stops + joins the reducer, clears `_pending`/`_dirty`). CORS `*`. Land
Step 3 first or stub the `node_reduce` imports.
*Files*: `src/goofi/node_data.py`.
*Acceptance*: framing unit tests round-trip at **100 B / 70 KB / 3 MB**; a test starts the
server, registers a fake node with one output slot, opens a raw-socket WS, `offer()`s a
`Data`, receives the reduced binary frame, sends a **masked** TEXT `{op:"view",spec,v}` and
asserts `viewspec_for` updates; `offer` twice rapidly ⇒ only the latest is reduced, no
orphaned `_dirty` entry; closing the socket decrements `viewer_count` to 0, runs
`evict_slot` (no pinned `Data`), and leaves no live reader thread; `_reset_for_tests` stops
the reducer thread (no live `goofi-data-reducer`) and frees the port (rebind succeeds).

**Step 3 — `node_reduce.py` + per-axis ViewSpec (Change C).**
*What*: `AxisSpec` + `ViewSpec` (frozen dataclasses, `axes`+`version`), `viewspec_from_dict`,
`reduce_for_view` (canonicalize axes, compose per-axis), `_apply_axis`, `_envelope`,
`_area_axis`/`_area_downscale_2d`, `_subsample_idx`, `_set_coord`, per-axis meta
co-reduction + `meta['reduced'][str(axis)]` reconstruction info per §6. Fail-open; never
mutate input; numpy-only.
*Files*: `src/goofi/node_reduce.py`.
*Acceptance*: unit tests — 1-D `{axis:-1,envelope}` → `2*W` envelope with co-reduced `dim0`
(`np.repeat(coord[centers],2)`) and `meta.reduced['0']={orig_len:N,method:'envelope'}`; 2-D
`(C,N)` with `[{0,subsample},{-1,envelope}]` → channel cap (dim0 gather) + envelope (dim1),
both coords correct, both meta keys present as `'0'`/`'1'` (negative axis canonicalized),
`Data(...)` constructs without tripping `data.py:104`; HD image `1920×1080×3` with
`[{0,area},{1,area}]` → `≤ max` (downscales at 1.5×/axis) and equals a true 2-D block mean
(separability test, max abs diff < 1e-6); envelope skip-if-ratio<2× returns input; subsample
`orig_coord` carried only for `orig_len ≤ 4096`; a deliberately broken input returns the
**unreduced** object.

**Step 4 — Wire node-side: bootstrap, advertise port, offer, idle-wake, teardown (Change A+B).**
*What*: init `self._log_port`/`self._data_port = None` (~`node.py:150`);
`register_node(node_id, self)` (returns port) in `__init__` (beside `node.py:173-174`); add
`"log_port"`/`"data_port"` to `_push_state` (`node.py:423`); the single `offer()` line after
`node.py:737` gated on `viewer_count > 0` (§5.3); `node._wake_processing()` on the `0→1`
viewer transition (done in the data server handler, §4.3); `unregister_node` in
`_messaging_loop` teardown (beside `node.py:490-491`). **No edits** to `_run_node_process`/
`_spawn_local`/`create_local`.
*Files*: `src/goofi/node.py`.
*Acceptance*: spawn a node with capture on; assert its `STATE_UPDATE` carries an int
`data_port` (and `log_port`); connect a raw WS to `ws://<host>:<data_port>/<node>/<slot>`,
send a per-axis ViewSpec, assert reduced frames arrive and shrink with the axis `max`; with
a real node consumer ALSO attached, assert node↔node iceoryx2 frames are still **full** size
**and unaffected** even if the reducer is forced to raise; viewing an idle free-running node
produces frames after connect.

**Step 5 — Manager host-scope export + bridge relay/snapshot for ports (Change B).**
*What*: `Manager.__init__` exports `GOOFI_BIND_HOST=bridge_host` in the non-headless block,
maps `0.0.0.0`→`127.0.0.1` for the printed URL, prints the LAN banner (§9.0); relay
`log_port`/`data_port` in `control.py:on_state` (beside `control.py:343`); add to
`schemas.describe_node_instance` (beside `schemas.py:99`); `node_log._LogServer` binds
`GOOFI_BIND_HOST` and `node_log.register_node` returns a port (§9.2).
*Files*: `src/goofi/manager.py`, `src/goofi/node_log.py`, `src/goofi/bridge/control.py`,
`src/goofi/bridge/schemas.py`.
*Acceptance*: a `/control` test client gets int `log_port` + `data_port` in both the `hello`
snapshot and a `state_update` event; a non-headless Manager binds node servers on
`GOOFI_BIND_HOST`; a headless Manager does **not** set the env (standalone node still binds
`127.0.0.1`); `git grep -n log_endpoint` returns nothing in `src/goofi`.

**Step 6 — Remove the manager-side data path.**
*What*: delete `bridge/data.py`; remove `DataHub` import/instantiation/route/`close_all`
from `server.py`; delete `NodeRef.set_data_handler`/`_data_pump`/`open_output_subscriber`/
`data_service_for` + data-pump fields + `terminate`'s `_data_waitset_dirty.set()`.
*Files*: `src/goofi/bridge/server.py`, `src/goofi/bridge/data.py` (deleted),
`src/goofi/node_helpers.py`.
*Acceptance*: `pytest tests/` green; `git grep -n set_data_handler src/goofi` empty; bridge
starts and serves `/control` + static.

**Step 7 — Frontend: types + graph merge + endpoint discovery + WS ownership + logStream (Change B).**
*What*: `control.ts` `log_port`/`data_port` (remove `log_endpoint`); `graph.svelte.ts`
merge; new `dataStream.svelte.ts` (reconcile, compose URL from `location.hostname` +
`data_port`, endpoint-moved compare, owns sockets); refactor `data.ts` to the
`connect(resolveEndpoint, getSpec)` helper shape (§8.3); `logStream.svelte.ts` composes its
SSE URL from `location.hostname` + `log_port` (§8.7). Grep the TS tree for `log_endpoint`
and fix every hit.
*Files*: `frontend/src/lib/api/control.ts`, `frontend/src/lib/stores/graph.svelte.ts`,
`frontend/src/lib/stores/dataStream.svelte.ts`, `frontend/src/lib/api/data.ts`,
`frontend/src/lib/stores/logStream.svelte.ts`.
*Acceptance*: `tsc` strict (no `any` in app code); a Playwright test loads a patch and
asserts the node's `data_port` appears in the store after the first `state_update`, a WS to
`${location.hostname}:<nodeport>` opens (DevTools network), and the log console still
streams (composed URL).

**Step 8 — Frontend: per-axis thalamus + inband ViewSpec + ViewerFeed lifecycle + ArrayViewer band + reduction-aware MetadataPanel (Change C).**
*What*: `thalamus.svelte.ts` (per-axis fold richest-wins/debounce/hysteresis/capacity per
§8.2); `ViewerFeed` consumer + need lifecycle (per-axis spec from capacity); `dataStream`
pushes folded per-axis spec via `data.ts.updateSpec`; ArrayViewer keys the band off the
**received** `meta.reduced[axis].method==='envelope'` and de-interleaves (§8.8);
MetadataPanel reconstructs and displays the **true original meta** (shape + coords from
`meta.reduced`, hides artifacts) per §8.9.
*Files*: `frontend/src/lib/stores/thalamus.svelte.ts`,
`frontend/src/lib/viewers/ViewerFeed.svelte`, `frontend/src/lib/viewers/ArrayViewer.svelte`,
`frontend/src/lib/stores/dataStream.svelte.ts` (spec push),
`frontend/src/lib/editor/MetadataPanel.svelte`.
*Acceptance*: `e2e/viewers.spec.ts` — open ArrayViewer on an Oscillator/PSD slot, assert a
reduced **band** render appears and reflows on canvas resize; two array viewers of differing
`max` on one slot both render a correct band from one folded `envelope` buffer; image viewer
downscales an HD frame; **MetadataPanel shows the original shape and coords** (not the
reduced `2*W` / co-reduced coords) for a reduced slot; no console errors.

**Step 9 — Stress + leak + co-location tests.**
*What*: open `test.gfi` (Oscillator + PSD + 8 Buffers + VideoStream) with 10+ viewers for
60 s; a Python test asserting `viewer_count` returns to 0 after a WS abruptly closes (leak
guard), the reader thread terminated, `_pending` holds no entry for the slot (`evict_slot`),
and the `goofi-data-reducer` thread stops on `_reset_for_tests`; a LOCAL-mode test asserting
the bridge TCPSite, node_log server, and node_data server bind distinct ports without error.
*Files*: `e2e/stress.spec.ts`; `tests/test_node_data.py`.
*Acceptance*: median ≥ 55 fps, no JS console errors, no Python tracebacks; `viewer_count`
returns to 0 on abrupt disconnect; no leaked reader or reducer thread; no pinned `Data`;
LOCAL 3-server bind clean; `pytest tests/` still green.

**Step 10 — Test fixture wiring.**
*What*: wherever the suite resets `node_log` (find via
`git grep -n "node_log._reset_for_tests"` — typically `tests/conftest.py`), also call
`node_data._reset_for_tests()` **and** `os.environ.pop("GOOFI_BIND_HOST", None)` so a
non-headless Manager test cannot bleed `0.0.0.0` into a later standalone-node test.
*Files*: the existing reset fixture (likely `tests/conftest.py`).
*Acceptance*: repeated full `pytest tests/` runs show no port/thread/env leakage (incl. the
reducer thread).

---

## 12. What Stays Untouched / Done When

### Stays untouched (HARD)

- `src/goofi/transport.py`, `src/goofi/codec.py`, the iceoryx2 setup.
- Node↔node publish work: `node.py:754-759` (now guarded on `subscriber_count > 0`, same
  work when subscribers exist, encoded **synchronously on the node thread**),
  `_ensure_output_endpoints`, `SUBSCRIBE_INPUT`.
- `_processing_loop`'s full-`Data` construction (`node.py:737`).
- `Manager` graph/link ownership, persistence, `/control` RPC + events, log SSE relay,
  static SPA serving + `no_cache` middleware. **`/control` stays same-origin-proxied.**
- `NodeRef.register_subscriber`/`unregister_subscriber` and node ctrl handling
  (`node.py:512-521`).
- `frontend/src/lib/api/frames.ts` (RAF coalescer signature) and
  `frontend/src/lib/codec/decode.ts` (already exposes `meta`; reduction needs no decode
  change).

### Done when

1. A browser viewing a slot connects **directly** to
   `ws://${location.hostname}:<nodeport>/<node>/<slot>` (verified in DevTools); the
   **manager is not** in the data path (`git grep -n set_data_handler src/goofi` empty;
   `bridge/data.py` deleted). The node servers bind `GOOFI_BIND_HOST` (default `0.0.0.0`,
   LAN-reachable); logs share the same host scope.
2. The node **reduces on the reducer thread before send**: a 44.1 kHz line slot ships
   ~`2*max` envelope frames (KB, not MB); an HD image slot ships a downscaled frame
   `≤ max` per axis (the 1× image threshold downscales 1920×1080→1280×720). The node's
   tick / node↔node output rate is **not slowed** by an attached viewer (Change A).
3. `viewer_count` gates production: a browser-only slot produces; closing the last viewer
   (incl. abrupt close, via `finally`) returns `viewer_count` to 0, runs `evict_slot`
   (no pinned `Data`), and the slot stops (unless a node consumer remains); a node consumer
   is never starved by a browser disconnect; the full-Data encode is **skipped** for
   viewer-only slots.
4. `log_port`/`data_port` ride `STATE_UPDATE` + snapshot exactly like the old log endpoint;
   the frontend composes URLs from `location.hostname`, reconnects across node restart (new
   port), and re-sends its per-axis ViewSpec.
5. Reduction is **fail-open**, **per-axis composed**, **meta co-reduces** (no `data.py:104`
   assertions), and is memory-safe across the reducer thread (a node mutating its returned
   array's contents cannot crash the reducer or corrupt/stall node↔node; the Change-A
   contract test passes; node↔node bytes are byte-for-byte unchanged).
6. **The meta inspector shows the TRUE original meta** for a reduced slot — original shape
   and original/index-range coords, with the `reduced` artifact hidden — off the single
   per-slot reduced stream (no second full stream).
7. `pytest tests/` (128 tests) green; RFC6455 framing unit tests pass at 100 B/70 KB/3 MB;
   `e2e/viewers.spec.ts` + `e2e/stress.spec.ts` pass (≥ 55 fps median, no console errors).
8. `node_data._reset_for_tests` parity — no leaked server/reader/**reducer** threads or
   bound ports across the suite; `GOOFI_BIND_HOST` does not leak between tests; LOCAL-mode
   3-server bind verified clean.