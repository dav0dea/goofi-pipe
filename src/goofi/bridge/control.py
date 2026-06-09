"""Control plane: JSON RPC + event stream over WS /control.

Wire format
-----------
Client → server (RPC requests):
    {"id": <int|null>, "op": "<name>", "payload": {...}}

Server → client:
    {"id": <int>, "result": ...}           # rpc reply
    {"id": <int>, "error": "<message>"}    # rpc error
    {"event": "<name>", "payload": {...}}  # broadcast event (no id)

Ops mirror the surface in CLAUDE.md §5. The hub also fans out manager
events (node added/removed, link added/removed, node state updates,
processing errors) to every connected client.
"""
from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from weakref import WeakSet

from aiohttp import WSMsgType, web

from goofi.bridge.schemas import (
    describe_node_instance,
    describe_params,
    list_node_types,
)
from goofi.message import Message, MessageType


class ControlHub:
    """One per BridgeServer. Holds connected clients + manager-side hooks."""

    def __init__(self, server) -> None:
        self.server = server
        self._clients: "WeakSet[web.WebSocketResponse]" = WeakSet()
        self._lock = asyncio.Lock()
        # Bookkeeping: when a node is added we register status-message
        # handlers on its NodeRef. Tracked so we can clean up on shutdown.
        self._wired_nodes: Set[str] = set()

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=30.0)
        await ws.prepare(request)
        async with self._lock:
            self._clients.add(ws)

        # Ensure every connected client gets node-status fan-out from any
        # nodes currently on the graph.
        for name in list(self.server.manager.nodes):
            self._wire_node_status(name)

        # Snapshot — let the client render before any events trickle in.
        try:
            await ws.send_json({"event": "hello", "payload": self._snapshot()})
        except Exception:
            return ws

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_text(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            async with self._lock:
                self._clients.discard(ws)
        return ws

    async def close_all(self) -> None:
        clients = list(self._clients)
        for ws in clients:
            try:
                await ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # broadcasts
    # ------------------------------------------------------------------

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        """Send to every connected client. Disconnections are silently ignored."""
        dead: List[web.WebSocketResponse] = []
        for ws in list(self._clients):
            try:
                if ws.closed:
                    dead.append(ws)
                    continue
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def broadcast_threadsafe(self, payload: Dict[str, Any]) -> None:
        """Manager-thread entry point to push an event to the bridge loop."""
        self.server.schedule(self.broadcast(payload))

    # ------------------------------------------------------------------
    # RPC dispatch
    # ------------------------------------------------------------------

    async def _handle_text(self, ws: web.WebSocketResponse, raw: str) -> None:
        import json

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        op = msg.get("op")
        msg_id = msg.get("id")
        payload = msg.get("payload") or {}

        try:
            result = await self._dispatch(op, payload)
        except Exception as e:
            traceback.print_exc()
            if msg_id is not None:
                await ws.send_json({"id": msg_id, "error": str(e)})
            return

        if msg_id is not None:
            await ws.send_json({"id": msg_id, "result": result})

    async def _dispatch(self, op: str, payload: Dict[str, Any]) -> Any:
        manager = self.server.manager
        if op == "list_nodes":
            return {"types": list_node_types()}
        if op == "list_graph":
            return self._snapshot()
        if op == "add_node":
            return await self._call_manager(
                manager.add_node,
                payload["type"],
                payload["category"],
                name=payload.get("name"),
                params=payload.get("params"),
                pos=tuple(payload.get("pos") or (0, 0)),
            )
        if op == "remove_node":
            await self._call_manager(manager.remove_node, payload["name"])
            return {"ok": True}
        if op == "add_link":
            await self._call_manager(
                manager.add_link,
                payload["node_out"],
                payload["node_in"],
                payload["slot_out"],
                payload["slot_in"],
            )
            return {"ok": True}
        if op == "remove_link":
            await self._call_manager(
                manager.remove_link,
                payload["node_out"],
                payload["node_in"],
                payload["slot_out"],
                payload["slot_in"],
            )
            return {"ok": True}
        if op == "update_param":
            ref = manager.nodes[payload["node"]]
            await self._call_manager(
                ref.update_param,
                payload["group"],
                payload["name"],
                payload["value"],
            )
            return {"ok": True}
        if op == "set_expression":
            ref = manager.nodes[payload["node"]]
            await self._call_manager(
                ref.set_expression,
                payload["group"],
                payload["name"],
                payload.get("expression"),
                bool(payload.get("expression_enabled", False)),
                bool(payload.get("expression_triggers_process", False)),
                bool(payload.get("expression_autoeval", False)),
            )
            return {"ok": True}
        if op == "set_node_pos":
            name = payload["name"]
            pos = payload["pos"]
            ref = manager.nodes[name]
            kwargs = dict(ref.gui_kwargs or {})
            kwargs["pos"] = list(pos)
            ref.gui_kwargs = kwargs
            manager.unsaved_changes = True
            await self.broadcast({"event": "node_moved", "payload": {"name": name, "pos": list(pos)}})
            return {"ok": True}
        if op == "set_node_viewers":
            # Per-output-slot view state (collapsed / kind / settings) the browser
            # keeps in sync so it round-trips into the .gfi on save. Soft UI state,
            # like the workspace layout, so it deliberately does not mark unsaved.
            ref = manager.nodes[payload["node"]]
            kwargs = dict(ref.gui_kwargs or {})
            kwargs["viewers"] = payload.get("viewers") or {}
            ref.gui_kwargs = kwargs
            return {"ok": True}
        if op == "set_layout":
            # The browser pushes its workspace layout into the running patch so
            # it survives reloads and lands in the .gfi on save. Layout is soft
            # UI state, so this deliberately does not mark the patch unsaved.
            manager.layout = payload.get("layout")
            return {"ok": True}
        if op == "save":
            path_arg = payload.get("path")
            overwrite = bool(payload.get("overwrite", False))
            # The browser ships its current workspace layout with the save so
            # it round-trips into the .gfi. Only overwrite when present so a
            # layout-less save can't wipe an existing one.
            if "layout" in payload:
                manager.layout = payload.get("layout")
            saved_path = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _save_and_return(manager, path_arg, overwrite)
            )
            return {"path": saved_path, "yaml": Path(saved_path).read_text(encoding="utf-8")}
        if op == "load":
            path = payload["path"]
            # Replace graph by clearing it first if needed.
            await self._call_manager(_replace_graph, manager, path)
            await self.broadcast({"event": "graph_replaced", "payload": self._snapshot()})
            return {"ok": True}
        if op == "load_text":
            # Frontend uploaded YAML content directly — write to a temp file
            # and load it. Used when the user clicks "Load" in the browser.
            import tempfile

            content = payload["content"]
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".gfi", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(content)
                tmp_path = tf.name
            try:
                await self._call_manager(_replace_graph, manager, tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            await self.broadcast({"event": "graph_replaced", "payload": self._snapshot()})
            return {"ok": True}
        if op == "ping":
            return {"pong": True}

        raise ValueError(f"unknown op: {op}")

    # ------------------------------------------------------------------
    # manager hooks — called from any thread
    # ------------------------------------------------------------------

    def on_node_added(self, name: str) -> None:
        self._wire_node_status(name)
        manager = self.server.manager
        try:
            ref = manager.nodes[name]
        except KeyError:
            return
        payload = describe_node_instance(name, ref)
        self.broadcast_threadsafe({"event": "node_added", "payload": payload})

    def on_node_removed(self, name: str) -> None:
        self._wired_nodes.discard(name)
        self.broadcast_threadsafe({"event": "node_removed", "payload": {"name": name}})

    def on_link_added(self, link: Dict[str, str]) -> None:
        self.broadcast_threadsafe({"event": "link_added", "payload": link})

    def on_link_removed(self, link: Dict[str, str]) -> None:
        self.broadcast_threadsafe({"event": "link_removed", "payload": link})

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, Any]:
        manager = self.server.manager
        nodes = []
        for name in list(manager.nodes):
            try:
                nodes.append(describe_node_instance(name, manager.nodes[name]))
            except Exception:
                traceback.print_exc()
        return {
            # Identifies the manager *process*. The browser keeps its tab open
            # across a backend restart and auto-reconnects; a changed id tells it
            # this is a fresh session (→ reset layout) rather than a transient
            # reconnect to the same one (→ keep the layout it already has).
            "instance_id": manager.instance_id,
            "nodes": nodes,
            "links": list(manager.links),
            "save_path": manager.save_path,
            "unsaved_changes": manager.unsaved_changes,
            "layout": manager.layout,
        }

    async def _call_manager(self, fn, *args, **kwargs) -> Any:
        """Run a (potentially blocking) manager call off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _wire_node_status(self, name: str) -> None:
        """Forward STATE_UPDATE / PROCESSING_ERROR from the node to clients."""
        if name in self._wired_nodes:
            return
        manager = self.server.manager
        try:
            ref = manager.nodes[name]
        except KeyError:
            return

        def on_state(noderef, message: Message):
            # Mirror the legacy bookkeeping: store the latest state on the
            # NodeRef (the messaging loop does this for the default path,
            # but a custom handler replaces it — re-do the work here).
            noderef.serialized_state = message.content
            noderef._first_state_event.set()
            try:
                noderef.params.update(message.content.get("params", {}))
            except Exception:
                pass
            self.broadcast_threadsafe(
                {
                    "event": "state_update",
                    "payload": {
                        "node": name,
                        "params": describe_params(noderef.params),
                        "output_subscribers": message.content.get("output_subscribers", {}),
                        # Advertise the node's SSE log endpoint as soon as it's known
                        # (first state push) so the frontend can subscribe peer-to-peer.
                        "log_endpoint": message.content.get("log_endpoint"),
                    },
                }
            )

        def on_error(noderef, message: Message):
            err = message.content.get("error")
            noderef.last_error = err
            self.broadcast_threadsafe(
                {"event": "error", "payload": {"node": name, "error": err}}
            )

        ref.set_message_handler(MessageType.STATE_UPDATE, on_state)
        ref.set_message_handler(MessageType.PROCESSING_ERROR, on_error)
        self._wired_nodes.add(name)


def _save_and_return(manager, path_arg, overwrite: bool) -> str:
    """Helper for the bridge save op: invokes manager.save and returns the
    final on-disk path (manager.save mutates manager.save_path)."""
    manager.save(filepath=path_arg, overwrite=overwrite)
    return manager.save_path or ""


def _replace_graph(manager, filepath: str) -> None:
    """Clear the current graph (if any) and load `filepath`.

    `Manager.load` refuses to load on top of an existing graph; the
    frontend's "Load" flow is destructive (replace) so we tear down
    every node first.
    """
    for n in list(manager.nodes):
        try:
            manager.remove_node(n, notify_gui=False)
        except Exception:
            traceback.print_exc()
    manager.load(filepath)
