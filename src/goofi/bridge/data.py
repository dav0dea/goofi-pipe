"""Data plane (Option C relay): per-(node, slot) reduced-frame relay.

A client viewing `node`'s `slot` opens
    ws://.../data/<node>/<slot>/<kind>

The producing **node** reduces the slot to its folded ViewSpec on a dedicated
reducer thread and publishes small GOOF frames on its `<dataservice>.view`
iceoryx2 service (see `goofi.node_viewer` / `goofi.node_reduce`). The manager
subscribes ONCE per (node, slot) via the NodeRef *viewer plane*
(`set_data_handler(..., view=True)`) and forwards those bytes to every connected
browser **verbatim** — there is **no** manager-side decode or re-encode. The
reduction (the ~1300× shrink) happens inside the node, before the cross-process
copy; the manager is a thin switchboard.

Each connection contributes a per-axis ViewSpec, seeded from its viewer `kind`
and overridable inband via a TEXT `{"op":"view","spec":{axes,version}}` message.
The hub folds the connected ViewSpecs richest-wins per (node, slot) and pushes the
fold to the node (`set_viewspec`). On the first viewer of a slot the node is
told to produce (REGISTER_VIEWER, via `set_data_handler(view=True)`); on the last
it stops (UNREGISTER_VIEWER).

Backpressure: each WS has a single-slot mailbox. New frames overwrite older
un-sent frames (latest-wins) so a slow client never stalls the producer.
"""
from __future__ import annotations

import asyncio
import functools
import json
from typing import Optional

from aiohttp import WSMsgType, web

from goofi.node_reduce import default_viewspec_for_kind, fold_viewspecs


class _SlotForwarder:
    """One per active WS connection. Owns a `_pending` slot + sender task, plus the
    per-axis ViewSpec this connection contributes to the slot's fold."""

    def __init__(self, ws: web.WebSocketResponse, loop: asyncio.AbstractEventLoop, spec: dict):
        self.ws = ws
        self.loop = loop
        self.spec = spec  # this connection's ViewSpec dict (seed from kind; inband override)
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
    """Fan-out for one (node, slot): one NodeRef view subscription → N forwarders.

    `dispatch` forwards the node-reduced bytes to every connected forwarder
    verbatim (no decode / no re-encode). The mux also folds the forwarders'
    ViewSpecs and pushes the fold to the node, so the node reduces exactly once
    to the richest representation any attached viewer needs.
    """

    def __init__(self, ref, slot: str):
        self.ref = ref
        self.slot = slot
        # Whole-tuple rebind on mutate → dispatch() (data-pump thread) always
        # reads a consistent snapshot without locking.
        self._forwarders: tuple = ()
        self._last_axes: Optional[list] = None

    def add(self, fwd) -> None:
        self._forwarders = (*self._forwarders, fwd)

    def remove(self, fwd) -> bool:
        self._forwarders = tuple(f for f in self._forwarders if f is not fwd)
        return not self._forwarders

    def dispatch(self, buf: bytes) -> None:
        """Forward a node-reduced GOOF frame to every connected forwarder verbatim.
        Runs on the NodeRef data-pump thread; `_forwarders` is read as a consistent
        whole-tuple snapshot."""
        for fwd in self._forwarders:
            fwd.push_threadsafe(buf)

    def push_spec_if_changed(self) -> None:
        """Re-fold the connected ViewSpecs and, if the reduction changed, push it to
        the node so it reduces to what the attached viewers actually need. Dedup on
        the `axes` only — the node ignores `version`, so a version-only bump (the
        client's monotonic counter) must not trigger a redundant ctrl publish.

        The push itself stays on the event loop: unlike the view-plane subscriber
        setup (which builds iceoryx2 services and is offloaded in the handler), this
        is one bounded, non-blocking latest-wins ctrl publish (loan + memcpy + notify)."""
        folded = fold_viewspecs([f.spec for f in self._forwarders])
        axes = folded.get("axes")
        if axes != self._last_axes:
            self._last_axes = axes
            try:
                self.ref.set_viewspec(self.slot, folded)
            except Exception:
                pass


class DataHub:
    def __init__(self, server) -> None:
        self.server = server
        self._muxes: dict = {}  # (uid, slot) -> _SlotMux
        self._lock = asyncio.Lock()

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        node = request.match_info["node"]
        slot = request.match_info["slot"]
        kind = request.match_info["kind"]

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
        # Seed this connection's ViewSpec from the viewer kind; the browser may
        # override it inband with a capacity-derived spec ({"op":"view"}).
        fwd = _SlotForwarder(ws, loop, default_viewspec_for_kind(kind))
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

                def on_frame(_noderef, _slot_name, buf, _mux=mux):
                    # The view-plane pump hands us the node-reduced GOOF bytes
                    # verbatim (raw=True); fan them out unchanged.
                    _mux.dispatch(buf)

                # set_data_handler(view=True) does blocking IPC (REGISTER_VIEWER +
                # iceoryx2 .view subscriber); run it off the event loop so it can't
                # stall other viewers' sends. Held under _lock so a concurrent
                # connect/disconnect for the same slot can't interleave.
                await loop.run_in_executor(
                    None,
                    functools.partial(ref.set_data_handler, slot, on_frame, raw=True, view=True),
                )
                self._muxes[key] = mux
            mux.add(fwd)
            mux.push_spec_if_changed()  # fold now includes this viewer

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Inband renegotiation: {"op":"view","spec":{axes,version}}.
                    try:
                        payload = json.loads(msg.data)
                    except Exception:
                        continue
                    if payload.get("op") == "view" and isinstance(payload.get("spec"), dict):
                        fwd.spec = payload["spec"]
                        async with self._lock:
                            mux.push_spec_if_changed()
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            async with self._lock:
                empty = mux.remove(fwd)
                if empty:
                    # Detach off the event loop too (blocking UNREGISTER_VIEWER +
                    # iceoryx2 teardown) so a slow disconnect can't stall other
                    # viewers; under _lock so a re-subscribe can't interleave.
                    try:
                        await loop.run_in_executor(None, ref.set_data_handler, slot, None)
                    except Exception:
                        pass
                    self._muxes.pop(key, None)
                else:
                    mux.push_spec_if_changed()  # re-fold without this viewer
            await fwd.close()
        return ws

    async def close_all(self) -> None:
        loop = asyncio.get_running_loop()
        # Hold _lock across the whole teardown: shutdown calls this while the server
        # still accepts WS connections, and handler() inserts a fresh mux under _lock
        # AFTER its awaits. Without the lock, a connect during one of our awaits would
        # add a mux past our snapshot, and the final clear() would drop it WITHOUT
        # UNREGISTER_VIEWER — re-leaking the very registration this method exists to
        # release. The awaits below (run_in_executor / ws.close) never re-enter _lock,
        # so there's no deadlock; a losing handler blocks until _muxes is cleared.
        async with self._lock:
            await self._close_all_locked(loop)

    async def _close_all_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        for mux in list(self._muxes.values()):
            # Tell the node its viewers are gone (UNREGISTER_VIEWER) and tear down the
            # .view subscriber — the per-connection finally does this, but a bulk
            # shutdown bypasses it, leaving the node reducing+publishing into a dead
            # subscriber. Off-loop, mirroring the handler's detach (blocking teardown).
            try:
                await loop.run_in_executor(None, mux.ref.set_data_handler, mux.slot, None)
            except Exception:
                pass
            for fwd in mux._forwarders:
                try:
                    await fwd.close()
                    if not fwd.ws.closed:
                        await fwd.ws.close()
                except Exception:
                    pass
        self._muxes.clear()
