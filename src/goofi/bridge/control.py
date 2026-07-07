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

from goofi.bridge import fsbrowse
from goofi.bridge.origin import reject_cross_origin
from goofi.bridge.schemas import (
    describe_instance,
    describe_node_instance,
    describe_params,
    list_node_types,
)
from goofi.manager import ROOT_ID
from goofi.message import Message, MessageType

# Control-plane protocol version, stamped into the `hello` handshake. The browser
# reconciles purely from echoed events, so a stale frontend/build/ against a newer
# backend would diverge SILENTLY; the client asserts this against its own compiled
# constant (frontend/src/lib/api/control.ts PROTOCOL_VERSION) and prompts a reload
# on mismatch. Bump BOTH sides together when the wire shape / reconciliation changes.
PROTOCOL_VERSION = 1


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

    async def handler(self, request: web.Request) -> web.StreamResponse:
        # Reject cross-origin upgrades before touching any state: the control plane
        # is unauthenticated and can spawn arbitrary-Python nodes (drive-by RCE).
        if (deny := reject_cross_origin(request, self.server.host)) is not None:
            return deny
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=30.0)
        await ws.prepare(request)
        async with self._lock:
            self._clients.add(ws)

        # Ensure every connected client gets node-status fan-out from any
        # nodes currently on the graph.
        for uid in list(self.server.manager.nodes):
            self._wire_node_status(uid)

        # One try/finally around BOTH the hello send and the message loop so the
        # _clients cleanup always runs — a hello-send failure must not early-return
        # past the discard and leak the ws into the broadcast set.
        try:
            # Snapshot — let the client render before any events trickle in. The
            # handshake also carries the protocol version (only on `hello`, not the
            # general snapshot) so the client can detect a stale-build skew.
            try:
                await ws.send_json(
                    {"event": "hello", "payload": {**self._snapshot(), "protocol_version": PROTOCOL_VERSION}}
                )
            except Exception:
                return ws
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

    def _splice_endpoint(self, manager, node: str, slot: str):
        """Translate a sub-patch boundary endpoint (instance id + boundary id) to
        the real inner (member uid, slot) so the flat link lands on the member. A
        normal node endpoint passes through unchanged. Raises (→ error reply) if
        the boundary is unwired — the inner target doesn't exist yet."""
        if node in getattr(manager, "_instances", {}):
            return manager.resolve_boundary(node, slot)
        return node, slot

    async def _dispatch(self, op: str, payload: Dict[str, Any]) -> Any:
        manager = self.server.manager
        if op == "list_nodes":
            return {"types": list_node_types(manager.node_specs)}
        if op == "list_graph":
            return self._snapshot()
        if op == "add_node":
            # ONE add path: scope selects where the node lands — a sub-patch instance
            # (member, mirrored across shared siblings) or, omitted, the ROOT graph. A
            # node is conceptually a member of some scope either way. `member_uid` carries
            # a node's ORIGINAL uid on redo-of-add / undo-of-delete so the re-created node
            # keeps its identity and the captured uid-keyed links + panels reconnect.
            return await self._call_manager(
                manager.add_node,
                payload["type"],
                payload["category"],
                scope=payload.get("inst_id") or ROOT_ID,
                name=payload.get("name"),
                params=payload.get("params"),
                pos=tuple(payload.get("pos") or (0, 0)),
                member_uid=payload.get("member_uid"),
            )
        if op == "remove_node":
            node = payload["node"]  # uid (real) or inst_id (sub-patch group node)
            # A sub-patch group node is virtual — deleting it removes the whole
            # sub-patch (members + links), mirroring node-delete semantics.
            if node in getattr(manager, "_instances", {}):
                await self._call_manager(manager.remove_instance, node)
            else:
                await self._call_manager(manager.remove_node, node)
            return {"ok": True}
        if op == "restart_node":
            # Crash recovery: respawn the node's process IN PLACE. The manager's
            # restart_node keeps the uid, display name, sub-patch membership, position and
            # links, and re-wires bridge status forwarding itself — so a sub-patch member
            # stays in its sub-patch (a remove+add would land it at ROOT and, for a SHARED
            # member, mirror-remove it across siblings).
            await self._call_manager(manager.restart_node, payload["node"])
            return {"ok": True}
        if op == "rename_node":
            # Set a node's mutable display name (safe — nothing keys on it).
            await self._call_manager(manager.rename_node, payload["node"], payload["name"])
            return {"ok": True}
        if op == "add_link":
            no, so = self._splice_endpoint(manager, payload["node_out"], payload["slot_out"])
            ni, si = self._splice_endpoint(manager, payload["node_in"], payload["slot_in"])
            await self._call_manager(manager.add_link, no, ni, so, si)
            return {"ok": True}
        if op == "remove_link":
            no, so = self._splice_endpoint(manager, payload["node_out"], payload["slot_out"])
            ni, si = self._splice_endpoint(manager, payload["node_in"], payload["slot_in"])
            await self._call_manager(manager.remove_link, no, ni, so, si)
            return {"ok": True}
        if op == "update_param":
            # Route through the manager so a shared sub-patch member mirrors the
            # edit across its sibling instances (strict mirror).
            await self._call_manager(
                manager.update_param,
                payload["node"],
                payload["group"],
                payload["name"],
                payload["value"],
            )
            return {"ok": True}
        if op == "set_expression":
            # Route through the manager so a shared sub-patch member mirrors the
            # binding across siblings + the definition (strict mirror), exactly like
            # update_param — never straight to the NodeRef.
            await self._call_manager(
                manager.set_expression,
                payload["node"],
                payload["group"],
                payload["name"],
                payload.get("expression"),
                bool(payload.get("expression_enabled", False)),
                bool(payload.get("expression_triggers_process", False)),
                bool(payload.get("expression_autoeval", False)),
            )
            return {"ok": True}
        if op == "set_node_pos":
            node = payload["node"]  # uid (real) or inst_id (sub-patch group node)
            pos = payload["pos"]
            # A sub-patch group node isn't a real node — persist its pos on the
            # instance record instead.
            if node in getattr(manager, "_instances", {}):
                manager._instances[node].pos = list(pos)
                manager.unsaved_changes = True
                await self.broadcast({"event": "node_moved", "payload": {"node": node, "pos": list(pos)}})
                return {"ok": True}
            # A real node — route through the manager so a shared sub-patch member
            # mirrors its position across sibling instances (strict mirror). Emit
            # a node_moved for every node the move touched (the node + siblings).
            changed = await self._call_manager(manager.set_node_pos, node, pos)
            for uid in changed:
                await self.broadcast({"event": "node_moved", "payload": {"node": uid, "pos": list(pos)}})
            return {"ok": True}
        if op == "set_node_viewers":
            # Per-output-slot view state (collapsed / kind / settings) the browser
            # keeps in sync so it round-trips into the .gfi on save. Soft UI state,
            # like the workspace layout, so it deliberately does not mark unsaved.
            # A sub-patch instance id has no NodeRef of its own: persist its
            # per-boundary-slot viewer state on the instance record so it round-trips
            # into the .gfi (backlog #17).
            if payload["node"] not in manager.nodes:
                inst = getattr(manager, "_instances", {}).get(payload["node"])
                if inst is not None:
                    inst.viewers = payload.get("viewers") or {}
                return {"ok": True}
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
            # Replace graph by clearing it first if needed. The teardown removes
            # old nodes with notify_gui=False (no on_node_removed), so reset the
            # status-wiring bookkeeping here — otherwise reloaded nodes that reuse
            # a display name are skipped by _wire_node_status and lose live
            # state/error forwarding (report B1).
            # Validate BEFORE any teardown or client resync, so a rejected load
            # (unknown type / bad version) is a true no-op — the graph AND the
            # session's console/viewport/selection/viewers are left untouched.
            await self._call_manager(manager.validate_loadable, path)
            self._wired_nodes.clear()
            try:
                await self._call_manager(_replace_graph, manager, path)
            finally:
                # Past validation, teardown has begun: resync every client to
                # backend truth even if the splice failed, so no browser renders
                # the destroyed graph. Stale data-plane muxes are dropped too.
                await self.server.data.close_all()
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
                # Validate the upload BEFORE any teardown/resync — a rejected
                # upload is a true no-op (see `load`).
                await self._call_manager(manager.validate_loadable, tmp_path)
                # Reset status-wiring bookkeeping before the destructive reload so
                # reloaded nodes reusing a display name get re-wired (report B1).
                self._wired_nodes.clear()
                try:
                    await self._call_manager(_replace_graph, manager, tmp_path)
                finally:
                    # Past validation: resync clients to backend truth even if the
                    # splice failed; drop stale data-plane muxes.
                    await self.server.data.close_all()
                    await self.broadcast({"event": "graph_replaced", "payload": self._snapshot()})
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return {"ok": True}
        if op == "group_nodes":
            inst_id = await self._call_manager(
                manager.group_nodes,
                payload["members"],
                payload.get("interface"),
                tuple(payload.get("pos") or (0, 0)),
                # Restore the prior instance id on redo-of-group / undo-of-expand so the
                # sub-patch keeps a stable identity across history (mirror of member_uid).
                inst_id=payload.get("inst_id"),
            )
            return {"inst_id": inst_id}
        if op == "expand_instance":
            restored = await self._call_manager(manager.expand_instance, payload["inst_id"])
            return {"restored": restored}
        if op == "duplicate_shared":
            # Promote an instance to shared (if needed) and spawn a mirror sibling.
            def_id = await self._call_manager(manager.share_instance, payload["inst_id"])
            new_inst = await self._call_manager(
                manager.instantiate_definition, def_id, tuple(payload.get("pos") or (0, 0))
            )
            return {"def_id": def_id, "inst_id": new_inst}
        if op == "make_unique":
            await self._call_manager(manager.make_unique, payload["inst_id"])
            return {"ok": True}
        if op == "re_share_instance":
            # Undo of make_unique: re-attach a unique instance to its ORIGINAL definition
            # (reuniting the strict-mirror family), instead of minting a fresh def + sibling.
            def_id = await self._call_manager(
                manager.re_share_instance, payload["inst_id"], payload["def_id"]
            )
            return {"def_id": def_id}
        if op == "add_boundary":
            bnd_id = await self._call_manager(
                manager.add_boundary,
                payload["inst_id"],
                payload["dir"],
                payload["dtype"],
                tuple(payload.get("pos") or (0, 0)),
            )
            return {"bnd_id": bnd_id}
        if op == "wire_boundary":
            await self._call_manager(
                manager.wire_boundary,
                payload["inst_id"],
                payload["bnd_id"],
                payload.get("inner_node"),
                payload.get("inner_slot"),
            )
            return {"ok": True}
        if op == "wire_boundary_to_leaf":
            created = await self._call_manager(
                manager.wire_boundary_to_leaf,
                payload["outer_inst"],
                payload["bnd"],
                payload["leaf_node"],
                payload["leaf_slot"],
            )
            # The auto-created intermediate (inst_id, bnd_id) pairs, so the client can
            # undo the whole chain (unwire the outer port + remove these) as one step.
            return {"created": [list(p) for p in created]}
        if op == "remove_boundary":
            await self._call_manager(manager.remove_boundary, payload["inst_id"], payload["bnd_id"])
            return {"ok": True}
        if op == "rename_boundary":
            await self._call_manager(
                manager.rename_boundary, payload["inst_id"], payload["bnd_id"], payload["name"]
            )
            return {"ok": True}
        if op == "set_boundary_pos":
            pos = list(payload["pos"])
            changed = await self._call_manager(
                manager.set_boundary_pos, payload["inst_id"], payload["bnd_id"], pos
            )
            for iid, bid in changed:
                await self.broadcast(
                    {"event": "boundary_moved", "payload": {"inst_id": iid, "bnd_id": bid, "pos": pos}}
                )
            return {"ok": True}
        if op == "list_dir":
            return await self._call_manager(fsbrowse.list_dir, payload.get("path"))
        if op == "list_examples":
            return fsbrowse.list_examples()
        if op == "serialize":
            yaml_text = await self._call_manager(manager.serialize_patch)
            return {"yaml": yaml_text}
        if op == "ping":
            return {"pong": True}

        raise ValueError(f"unknown op: {op}")

    # ------------------------------------------------------------------
    # manager hooks — called from any thread
    # ------------------------------------------------------------------

    def on_node_added(self, uid: str) -> None:
        self._wire_node_status(uid)
        manager = self.server.manager
        try:
            ref = manager.nodes[uid]
        except KeyError:
            return
        payload = describe_node_instance(uid, ref)
        self.broadcast_threadsafe({"event": "node_added", "payload": payload})

    def on_node_removed(self, uid: str, membership: Optional[Dict[str, Any]] = None) -> None:
        # `membership` is the scope the node was a member of ({instance, local_name} —
        # ROOT for a top-level node), captured by the manager BEFORE detach. The frontend
        # drops the uid from that instance's members map (the index it renders the scope's
        # children from), so an incremental remove keeps the owning scope in sync without a
        # snapshot.
        self._wired_nodes.discard(uid)
        self.broadcast_threadsafe(
            {"event": "node_removed", "payload": {"node": uid, "membership": membership}}
        )

    def on_link_added(self, link: Dict[str, str]) -> None:
        self.broadcast_threadsafe({"event": "link_added", "payload": link})

    def on_link_removed(self, link: Dict[str, str]) -> None:
        self.broadcast_threadsafe({"event": "link_removed", "payload": link})

    def on_node_renamed(self, uid: str, name: str) -> None:
        # Identity is the uid (unchanged); only the display name moved. The status
        # fan-out is keyed by uid, so nothing to re-wire — just notify the browser.
        self.broadcast_threadsafe({"event": "node_renamed", "payload": {"node": uid, "name": name}})

    def on_subpatch_changed(self) -> None:
        # Group/expand/add-member/share rewrites the node SET structurally and is the
        # ONLY event it fires — members spawn (add_member_node, shared-sibling mirror)
        # and tear down (remove_instance) silently, with no on_node_added/on_node_removed.
        # So reconcile status forwarding HERE: wire any new member's STATE_UPDATE / error
        # / stats handlers, else a node created inside a sub-patch never echoes a param
        # edit and its panel snaps back to the creation default; and drop bookkeeping for
        # members that vanished, else a reused uid would early-return unwired. Idempotent
        # + uid-keyed, so this is a cheap sweep over the current node set.
        manager = self.server.manager
        live = set(manager.nodes)
        for uid in self._wired_nodes - live:
            self._wired_nodes.discard(uid)
        for uid in live:
            self._wire_node_status(uid)
        # Push a fresh snapshot so clients re-sync without a wholesale re-fit.
        self.broadcast_threadsafe({"event": "subpatch_changed", "payload": self._snapshot()})

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, Any]:
        manager = self.server.manager
        nodes = []
        for uid in list(manager.nodes):
            try:
                nodes.append(describe_node_instance(uid, manager.nodes[uid]))
            except Exception:
                traceback.print_exc()
        # Attribute each errored node to its ancestor instances in ONE pass (the first
        # errored descendant, in node order, wins), instead of re-scanning all nodes per
        # instance inside describe_instance.
        err_by_inst: Dict[str, Any] = {}
        # Snapshot the keys (like the node loop above) — a structural RPC on an executor
        # thread can mutate manager.nodes while this runs on the event loop; iterating the
        # live dict would raise 'dictionary changed size during iteration' and abort the
        # whole snapshot. Guard the per-node read too, in case the node is gone by now.
        for u in list(manager.nodes):
            try:
                err = manager.nodes[u].last_error
            except KeyError:
                continue
            if not err:
                continue
            for inst_id in manager._ancestor_instances(u):
                err_by_inst.setdefault(inst_id, err)
        instances: Dict[str, Any] = {}
        for iid, inst in getattr(manager, "_instances", {}).items():
            # ROOT ships too: it's a real scope (parent=None, no boundaries) whose members
            # are every top-level entity, so the editor renders the root canvas as
            # childrenOfScope(ROOT_ID) with no null special case. It is never synthesized
            # as a selectable group node (the frontend filters ROOT_ID out of nodeById).
            try:
                instances[iid] = describe_instance(manager, iid, inst, error=err_by_inst.get(iid))
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
            # Sub-patch instances (flatten-at-runtime): the editor renders these as
            # collapsible group nodes; each is a server-computed record the frontend
            # mirrors (see describe_instance). Guarded per-instance, like the per-node
            # loop above, so one broken instance can't drop the whole snapshot.
            "instances": instances,
            "save_path": manager.save_path,
            "unsaved_changes": manager.unsaved_changes,
            "layout": manager.layout,
        }

    async def _call_manager(self, fn, *args, **kwargs) -> Any:
        """Run a (potentially blocking) manager call off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _wire_node_status(self, uid: str) -> None:
        """Forward STATE_UPDATE / PROCESSING_ERROR from the node to clients. Keyed
        by the node's stable uid, so it survives a rename and re-points on restart."""
        if uid in self._wired_nodes:
            return
        manager = self.server.manager
        try:
            ref = manager.nodes[uid]
        except KeyError:
            return

        # These run AFTER the NodeRef messaging loop has already applied the push
        # to its cache (the single apply site, NodeRef._dispatch_status), so they
        # read fresh node-authoritative values and only fan them out to browsers.
        def on_state(noderef, message: Message):
            self.broadcast_threadsafe(
                {
                    "event": "state_update",
                    "payload": {
                        "node": uid,
                        "params": describe_params(noderef.params),
                        "stage": noderef.stage,
                        # The node's current error rides the state plane so a lost first
                        # PROCESSING_ERROR still reaches the browser and a healthy
                        # respawn's null lifts the stale chip (the manager-side diff is
                        # silent on a fresh ref whose last_error is already None).
                        "error": noderef.last_error,
                        "output_subscribers": message.content.get("output_subscribers", {}),
                        # Advertise the node's SSE log endpoint as soon as it's known
                        # (first state push) so the frontend can subscribe peer-to-peer.
                        "log_endpoint": message.content.get("log_endpoint"),
                    },
                }
            )
            # Refresh each ANCESTOR sub-patch instance's error on every push. The
            # diff-gated _broadcast_error_fanout skips a fresh ref's None==None
            # (a restarted member), so without this a collapsed group node keeps a
            # stale error chip after the member recovers. Empty for top-level nodes.
            manager = self.server.manager
            for inst_id in manager._ancestor_instances(uid):
                self.broadcast_threadsafe(
                    {"event": "error", "payload": {"node": inst_id, "error": manager._instance_error(inst_id)}}
                )

        def on_error(noderef, message: Message):
            self._broadcast_error_fanout(uid, message.content.get("error"))

        def on_stats(noderef, message: Message):
            self.broadcast_threadsafe(
                {"event": "node_stats", "payload": {"node": uid, "stats": message.content.get("stats")}}
            )

        ref.set_message_handler(MessageType.STATE_UPDATE, on_state)
        ref.set_message_handler(MessageType.PROCESSING_ERROR, on_error)
        ref.set_message_handler(MessageType.NODE_STATS, on_stats)
        self._wired_nodes.add(uid)

    def rewire_node_status(self, uid: str) -> None:
        """Re-point the status handlers at a node's NEW NodeRef after a restart.
        The old handlers lived on the now-dead ref; drop the bookkeeping and wire
        the fresh one so its first healthy push (which clears the crash chip) flows."""
        self._wired_nodes.discard(uid)
        self._wire_node_status(uid)

    def on_node_crashed(self, uid: str, exitcode, restart_count: int) -> None:
        """Surface a crashed node process to the browser as a DISTINCT crash state
        (not a code error): a process crash is transient and auto-recovering. The
        frontend clears it on the respawned node's first healthy state push."""
        self.broadcast_threadsafe(
            {"event": "node_crashed", "payload": {"node": uid, "exitcode": exitcode, "restarts": restart_count}}
        )

    def _broadcast_error_fanout(self, uid: str, error: Optional[str]) -> None:
        """Broadcast a node's error on the error channel, keyed by its uid AND by
        each ANCESTOR sub-patch instance. A collapsed sub-patch mirrors its
        first-errored-descendant on the snapshot field `inst.error`, which only
        recomputes on a structural resync — so a live member error (or its
        clearing) must refresh each ancestor. The frontend's `error` handler
        updates a node OR an instance in place, and ingests member errors to the
        console. Shared by the live PROCESSING_ERROR path and terminal bootstrap
        errors (on_node_stage), which otherwise reach neither."""
        self.broadcast_threadsafe({"event": "error", "payload": {"node": uid, "error": error}})
        manager = self.server.manager
        for inst_id in manager._ancestor_instances(uid):
            self.broadcast_threadsafe(
                {"event": "error", "payload": {"node": inst_id, "error": manager._instance_error(inst_id)}}
            )

    def on_node_stage(self, uid: str, stage: str, error: Optional[str] = None) -> None:
        """Broadcast a lifecycle-stage transition (creating/setup/ready/error).
        `error` carries the bootstrap traceback for the terminal error stage."""
        payload: Dict[str, Any] = {"node": uid, "stage": stage}
        if error is not None:
            payload["error"] = error
        self.broadcast_threadsafe({"event": "node_stage", "payload": payload})
        # A terminal bootstrap error reaches the browser ONLY via node_stage; run
        # the standard error fan-out too so the console + ancestor sub-patch
        # instances see it (the child never got to send a PROCESSING_ERROR).
        if error is not None:
            self._broadcast_error_fanout(uid, error)


def _save_and_return(manager, path_arg, overwrite: bool) -> str:
    """Helper for the bridge save op: invokes manager.save and returns the
    final on-disk path (manager.save mutates manager.save_path)."""
    manager.save(filepath=path_arg, overwrite=overwrite)
    return manager.save_path or ""


def _replace_graph(manager, filepath: str) -> None:
    """Validate, then destructively replace the current graph with `filepath`.

    Validation runs BEFORE any teardown, so a rejected load (unknown node type,
    unsupported version, missing file) leaves the current graph fully intact.
    `clear_graph` (not a per-node remove loop) also wipes sub-patch instances,
    definitions, and membership — else the old graph's instances survive as
    phantom empty group nodes.
    """
    manager.validate_loadable(filepath)  # raises here -> current graph untouched
    manager.clear_graph()
    manager.load(filepath)
