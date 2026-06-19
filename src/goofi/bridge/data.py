"""Data plane: per-(node, slot) binary WebSocket.

A client wishing to view `node`'s `slot` output opens
    ws://.../data/<node>/<slot>

The hub registers a data handler on the corresponding NodeRef
(`NodeRef.set_data_handler`) and forwards every encoded `Data` frame
verbatim — the browser holds the matching codec, so no transcoding
happens server-side.

Backpressure: each WS has a single-slot mailbox. New frames overwrite
older un-sent frames (latest-wins) so a slow client never stalls the
producer or piles up memory.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from aiohttp import WSMsgType, web

from goofi.codec import encode_data_into, prepare_encode


class _SlotForwarder:
    """One per active WS connection. Owns a `_pending` slot + sender task."""

    def __init__(self, ws: web.WebSocketResponse, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self._pending: Optional[bytes] = None
        self._dirty = asyncio.Event()
        self._closed = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = self.loop.create_task(self._run())

    def push_threadsafe(self, frame: bytes) -> None:
        """Called from the NodeRef data-pump thread."""

        def _set():
            if self._closed:
                return
            self._pending = frame  # overwrite — latest wins
            self._dirty.set()

        try:
            self.loop.call_soon_threadsafe(_set)
        except RuntimeError:
            # Loop is closed; drop frame silently.
            pass

    async def close(self) -> None:
        self._closed = True
        self._dirty.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self) -> None:
        try:
            while not self._closed:
                await self._dirty.wait()
                self._dirty.clear()
                if self._pending is None:
                    continue
                frame = self._pending
                self._pending = None
                if self.ws.closed:
                    return
                try:
                    await self.ws.send_bytes(frame)
                except (ConnectionResetError, RuntimeError):
                    return
        except asyncio.CancelledError:
            pass


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
        # A sub-patch output is viewed exactly like a node output: the client opens
        # /data/<instId>/<boundary>. Splice that to the inner member's real (node,
        # slot) so the same streaming path serves both. Unwired/unknown boundaries
        # close terminally (the browser won't retry a 4000-range close).
        if node in getattr(manager, "_instances", {}):
            try:
                node, slot = manager.resolve_boundary(node, slot)
            except (KeyError, ValueError):
                await ws.close(code=4004, message=b"unwired boundary")
                return ws

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

        # Key the mux by the node's STABLE member_uid, not its display name, so a
        # rename (sub-patch group/expand renames in place) doesn't spawn a second
        # mux that evicts the first's handler — viewers keep streaming across it.
        uid = ref.member_uid or node
        key = (uid, slot)
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
                # We don't expect inbound traffic; drain to detect close.
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
        for mux in list(self._muxes.values()):
            for fwd in mux._forwarders:
                try:
                    await fwd.close()
                    if not fwd.ws.closed:
                        await fwd.ws.close()
                except Exception:
                    pass
        self._muxes.clear()
