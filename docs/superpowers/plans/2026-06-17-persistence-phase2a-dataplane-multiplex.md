# Phase 2a (data plane) — Multiplex viewers per slot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Allow N simultaneous viewers (WebSocket clients) on the same `(node, slot)` to each receive every frame — fixing the current bug where a second viewer evicts the first's data handler.

**Architecture:** The bridge `DataHub` gains a per-`(node, slot)` multiplexer (`_SlotMux`). It registers exactly **one** `NodeRef.set_data_handler` per slot, encodes each frame once, and fans it out to every connected `_SlotForwarder`. The handler is unregistered only when the last forwarder for that slot closes. No format/manager/frontend change. (Spec §2.11; the `/data/by-uid` re-key is deferred to the grouping chunk.)

**Tech Stack:** Python 3.12 + aiohttp; pytest.

## Global Constraints
- Encode each frame **once** per `(node, slot)` (not per viewer) — it's the hot path.
- Fan-out reads must be thread-safe: `dispatch()` runs on the NodeRef data-pump thread; add/remove run on the asyncio loop. Use whole-tuple rebind (atomic in CPython) so `dispatch` always sees a consistent snapshot.
- Run unit tests: `.venv/bin/python -m pytest tests/test_datahub_mux.py -p no:cacheprovider -q`

## File Structure
- Modify: `src/goofi/bridge/data.py` — add `_SlotMux`; rework `DataHub` to multiplex.
- Create: `tests/test_datahub_mux.py` — unit tests for fan-out + refcount.
- Create: `e2e/test_data_multiplex.py` — integration: two raw WS on one slot both receive frames (gitignored).

---

## Task 1: `_SlotMux` fan-out + DataHub multiplex

**Files:**
- Modify: `src/goofi/bridge/data.py`
- Test: `tests/test_datahub_mux.py`

**Interfaces:**
- Produces: `_SlotMux(ref, slot)` with `add(fwd)`, `remove(fwd) -> bool` (True when empty), `dispatch(frame: bytes) -> None`.

- [ ] **Step 1: Write the failing unit test**

```python
# tests/test_datahub_mux.py
"""Unit tests for the data-plane per-slot multiplexer."""
from goofi.bridge.data import _SlotMux


class _FakeFwd:
    def __init__(self):
        self.frames = []

    def push_threadsafe(self, frame: bytes) -> None:
        self.frames.append(frame)


def test_dispatch_fans_out_to_all_forwarders():
    mux = _SlotMux(ref=None, slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)
    mux.dispatch(b"frame")
    assert a.frames == [b"frame"]
    assert b.frames == [b"frame"]


def test_remove_keeps_others_and_reports_empty():
    mux = _SlotMux(ref=None, slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)
    assert mux.remove(a) is False  # b still connected
    mux.dispatch(b"y")
    assert a.frames == []  # removed → no frames
    assert b.frames == [b"y"]
    assert mux.remove(b) is True  # last one out → empty
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_datahub_mux.py -p no:cacheprovider -q`
Expected: FAIL — `ImportError: cannot import name '_SlotMux'`

- [ ] **Step 3: Implement `_SlotMux` and rework `DataHub`**

Add `_SlotMux` after the `_SlotForwarder` class in `data.py`:

```python
class _SlotMux:
    """Fan-out for one (node, slot): one NodeRef data-handler → N forwarders.

    NodeRef.set_data_handler is single-callback-per-slot and evicting, so the
    bridge multiplexes here: ONE handler per (node, slot), each frame encoded
    once and dispatched to every connected forwarder. The handler is dropped
    only when the last forwarder closes.
    """

    def __init__(self, ref, slot: str):
        self.ref = ref
        self.slot = slot
        # Whole-tuple rebind on mutate → dispatch() (data-pump thread) always
        # reads a consistent snapshot without locking.
        self._forwarders: tuple = ()

    def add(self, fwd) -> None:
        self._forwarders = (*self._forwarders, fwd)

    def remove(self, fwd) -> bool:
        self._forwarders = tuple(f for f in self._forwarders if f is not fwd)
        return not self._forwarders

    def dispatch(self, frame: bytes) -> None:
        for fwd in self._forwarders:
            fwd.push_threadsafe(frame)
```

Replace the `DataHub` class body with the multiplexing version:

```python
class DataHub:
    def __init__(self, server) -> None:
        self.server = server
        self._muxes: dict = {}  # (node, slot) -> _SlotMux
        self._lock = asyncio.Lock()

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        node = request.match_info["node"]
        slot = request.match_info["slot"]

        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=30.0)
        await ws.prepare(request)

        manager = self.server.manager
        if node not in manager.nodes:
            await ws.close(code=4004, message=b"unknown node")
            return ws
        ref = manager.nodes[node]
        if slot not in ref.output_slots:
            await ws.close(code=4004, message=b"unknown slot")
            return ws

        loop = asyncio.get_running_loop()
        fwd = _SlotForwarder(ws, loop)
        fwd.start()

        key = (node, slot)
        async with self._lock:
            mux = self._muxes.get(key)
            if mux is None:
                mux = _SlotMux(ref, slot)
                self._muxes[key] = mux

                def on_frame(_noderef, _slot_name, data, _mux=mux):
                    try:
                        size, meta_bytes = prepare_encode(data)
                        buf = bytearray(size)
                        encode_data_into(data, memoryview(buf), meta_bytes=meta_bytes)
                        _mux.dispatch(bytes(buf))
                    except Exception:
                        import traceback

                        traceback.print_exc()

                ref.set_data_handler(slot, on_frame)
            mux.add(fwd)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            async with self._lock:
                empty = mux.remove(fwd)
                if empty:
                    try:
                        ref.set_data_handler(slot, None)
                    except Exception:
                        pass
                    self._muxes.pop(key, None)
            await fwd.close()
        return ws

    async def close_all(self) -> None:
        muxes = list(self._muxes.values())
        for mux in muxes:
            for fwd in mux._forwarders:
                try:
                    await fwd.close()
                    if not fwd.ws.closed:
                        await fwd.ws.close()
                except Exception:
                    pass
        self._muxes.clear()
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_datahub_mux.py -p no:cacheprovider -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goofi/bridge/data.py tests/test_datahub_mux.py
git commit -m "fix(bridge): multiplex data viewers per slot (one handler, N forwarders)"
```

---

## Task 2: Integration test — two viewers on one slot

**Files:**
- Create: `e2e/test_data_multiplex.py` (gitignored; runs against the live bridge fixture)

- [ ] **Step 1: Write the integration test**

```python
# e2e/test_data_multiplex.py
"""Two data-plane WS on the same (node, slot) must BOTH receive frames.

Regresses the single-callback-eviction bug: before the multiplex fix the
second subscriber evicted the first, so only one viewer got data.
"""
from __future__ import annotations

import asyncio
import json

import aiohttp


def test_two_viewers_one_slot_both_receive(bridge: str):
    async def run() -> None:
        ws_base = bridge.replace("http", "ws", 1)
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"{ws_base}/control") as ctl:
                await ctl.receive()  # hello
                await ctl.send_json(
                    {"id": 1, "op": "add_node",
                     "payload": {"type": "Oscillator", "category": "inputs"}}
                )
                name = None
                while name is None:
                    m = await asyncio.wait_for(ctl.receive(), timeout=10)
                    d = json.loads(m.data)
                    if d.get("id") == 1:
                        name = d["result"]

                async with s.ws_connect(f"{ws_base}/data/{name}/out") as w1, \
                        s.ws_connect(f"{ws_base}/data/{name}/out") as w2:
                    f1 = await asyncio.wait_for(w1.receive(), timeout=10)
                    f2 = await asyncio.wait_for(w2.receive(), timeout=10)
                    assert f1.type == aiohttp.WSMsgType.BINARY
                    assert f2.type == aiohttp.WSMsgType.BINARY

    asyncio.run(run())
```

- [ ] **Step 2: Run the integration test**

Run: `.venv/bin/python -m pytest e2e/test_data_multiplex.py -p no:cacheprovider -q`
Expected: PASS (1 passed). (Boots a real Manager via the `bridge` fixture; Oscillator autotriggers so `out` produces frames.)

- [ ] **Step 3: (e2e is gitignored — nothing to commit)**

---

## Self-Review
- Spec coverage: §2.11 "DataHub must hold ONE handler per (uid, slot) and fan out to a set of forwarders" — implemented per `(node, slot)` for this chunk (uid re-key deferred with grouping). ✓
- Placeholder scan: none. ✓
- Type consistency: `_SlotMux.add/remove/dispatch` used identically in `DataHub.handler`/`close_all`. ✓
- Final: run `.venv/bin/python -m pytest tests/ -p no:cacheprovider -q` to confirm no regression.
