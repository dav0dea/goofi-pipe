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
from typing import Optional, Set

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


class DataHub:
    def __init__(self, server) -> None:
        self.server = server
        self._active: "Set[_SlotForwarder]" = set()
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

        def on_frame(_noderef, _slot_name, data):
            try:
                size, meta_bytes = prepare_encode(data)
                buf = bytearray(size)
                encode_data_into(data, memoryview(buf), meta_bytes=meta_bytes)
                fwd.push_threadsafe(bytes(buf))
            except Exception:
                import traceback

                traceback.print_exc()

        ref.set_data_handler(slot, on_frame)
        async with self._lock:
            self._active.add(fwd)

        try:
            async for msg in ws:
                # We don't expect inbound traffic; drain to detect close.
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            try:
                ref.set_data_handler(slot, None)
            except Exception:
                pass
            await fwd.close()
            async with self._lock:
                self._active.discard(fwd)
        return ws

    async def close_all(self) -> None:
        for fwd in list(self._active):
            try:
                await fwd.close()
                if not fwd.ws.closed:
                    await fwd.ws.close()
            except Exception:
                pass
