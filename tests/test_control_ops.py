"""The bridge control ops added for Phase-1 save/load (no live socket)."""
import asyncio
import types
from pathlib import Path

from goofi.bridge.control import ControlHub
from goofi.bridge.data import DataHub
from goofi.message import MessageType

from .test_manager import _bare_manager


def _hub(manager=None) -> ControlHub:
    # ControlHub needs `server.manager` for these ops + `server.data` (the load path
    # tears down the data plane). No real socket; `schedule` swallows the coroutine.
    server = types.SimpleNamespace(manager=manager, schedule=lambda coro: coro.close())
    server.data = DataHub(server)
    return ControlHub(server)


def test_list_dir_op(tmp_path: Path):
    (tmp_path / "x.gfi").write_text("x")
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_dir", {"path": str(tmp_path)}))
    assert res["path"] == str(tmp_path.resolve())
    assert any(e["name"] == "x.gfi" for e in res["entries"])


def test_list_examples_op():
    hub = _hub()
    res = asyncio.run(hub._dispatch("list_examples", {}))
    assert "entries" in res


def test_serialize_op_calls_manager():
    class FakeMgr:
        def serialize_patch(self):
            return "nodes: {}\nlinks: []\n"

    hub = _hub(FakeMgr())
    res = asyncio.run(hub._dispatch("serialize", {}))
    assert res["yaml"].startswith("nodes:")


def test_add_node_dispatch_forwards_member_uid():
    """Redo-of-add and undo-of-delete re-create a node with its ORIGINAL uid
    (sent as `member_uid`) so uid-keyed links and panel bindings reconnect. The
    bridge must forward member_uid to the manager; dropping it mints a fresh uid
    and orphans the captured links (KeyError on the replayed add_link)."""
    manager = _bare_manager()
    manager._layout = None
    try:
        hub = _hub(manager)
        want = "deadbeef0001"
        got = asyncio.run(
            hub._dispatch("add_node", {"type": "Oscillator", "category": "inputs", "member_uid": want})
        )
        assert got == want, f"member_uid was not honored (bridge dropped it): {got!r}"
        assert want in manager.nodes
    finally:
        manager.terminate(notify_gui=False)


def test_load_rewires_state_forwarding_for_reused_names(tmp_path: Path):
    """Report B1: a destructive reload tears down old nodes with notify_gui=False
    (so _wired_nodes is never cleared), then re-adds nodes with the SAME display
    names — _wire_node_status then early-returns and the reloaded nodes never get
    their STATE_UPDATE/PROCESSING_ERROR handlers. Live forwarding silently dies."""
    manager = _bare_manager()
    manager._layout = None
    try:
        hub = _hub(manager)
        manager._bridge = types.SimpleNamespace(control=hub)

        osc = manager.add_node("Oscillator", "inputs")
        assert MessageType.STATE_UPDATE in manager.nodes[osc].callbacks
        assert osc in hub._wired_nodes

        gfi = tmp_path / "patch.gfi"
        manager.save(str(gfi))

        # Destructive reload of the same patch reuses the node's display name.
        asyncio.run(hub._dispatch("load", {"path": str(gfi)}))

        assert osc in manager.nodes, "node not reloaded under the same name"
        # The reloaded node's NEW NodeRef must have its forwarding handlers back.
        assert MessageType.STATE_UPDATE in manager.nodes[osc].callbacks, "state forwarding lost after reload (B1)"
        assert MessageType.PROCESSING_ERROR in manager.nodes[osc].callbacks
    finally:
        manager.terminate(notify_gui=False)


def test_snapshot_error_pass_tolerates_concurrent_node_removal():
    """_snapshot's error-attribution pass must iterate a SNAPSHOT of manager.nodes, not the
    live dict: a structural RPC on another executor thread can remove a node mid-iteration,
    and iterating the live dict would raise 'dictionary changed size during iteration',
    aborting the whole snapshot (and suppressing graph_replaced on the load path)."""
    mgr = _bare_manager(use_multiprocessing=False)
    mgr._layout = None
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.nodes[a].last_error = "boom-a"
        mgr.nodes[b].last_error = "boom-b"
        hub = _hub(mgr)

        # Simulate a concurrent structural mutation: the first ancestor lookup in the error
        # pass drops the OTHER node, so a live-dict iteration would blow up on the next step.
        orig = mgr._ancestor_instances
        state = {"evicted": False}

        def _evict_then_delegate(uid):
            if not state["evicted"]:
                state["evicted"] = True
                mgr.nodes._nodes.pop(b if uid == a else a, None)
            return orig(uid)

        mgr._ancestor_instances = _evict_then_delegate

        snap = hub._snapshot()  # must NOT raise
        assert "instances" in snap and "nodes" in snap
    finally:
        mgr.terminate(notify_gui=False)


def test_load_tears_down_the_data_plane(tmp_path: Path):
    """A destructive reload replaces every node, so every data-plane mux is left caching
    a dead NodeRef. The load op must tear the data plane down (close_all) so viewers
    reconnect against the new graph instead of binding to a dead ref / reused-uid mux."""
    manager = _bare_manager()
    manager._layout = None
    try:
        hub = _hub(manager)
        closed = []

        class _SpyData:
            async def close_all(self):
                closed.append(True)

        hub.server.data = _SpyData()
        manager._bridge = types.SimpleNamespace(control=hub)
        manager.add_node("Oscillator", "inputs")
        gfi = tmp_path / "patch.gfi"
        manager.save(str(gfi))

        asyncio.run(hub._dispatch("load", {"path": str(gfi)}))

        assert closed, "load did not tear down the data-plane muxes"
    finally:
        manager.terminate(notify_gui=False)


def test_wire_boundary_to_leaf_dispatch_returns_created_chain():
    """The bridge surfaces the auto-chain op and returns the auto-created intermediate
    (inst_id, bnd_id) pairs so the frontend can undo the whole chain as one step."""
    from .test_manager import _build_grouped_graph, _member

    manager = _bare_manager(use_multiprocessing=False)
    manager._layout = None
    try:
        hub = _hub(manager)
        osc, inner = _build_grouped_graph(manager)
        s1 = _member(manager, inner, "select1")
        out_slot = list(manager.nodes[s1].output_slots)[0]
        outer = manager.group_nodes([inner])
        outer_bnd = manager.add_boundary(outer, "out", "ARRAY")
        res = asyncio.run(
            hub._dispatch(
                "wire_boundary_to_leaf",
                {"outer_inst": outer, "bnd": outer_bnd, "leaf_node": s1, "leaf_slot": out_slot},
            )
        )
        assert len(res["created"]) == 1  # one auto-created intermediate boundary
        assert manager.resolve_boundary(outer, outer_bnd) == (s1, out_slot)
    finally:
        manager.terminate(notify_gui=False)


def test_member_node_added_inside_subpatch_gets_status_wired():
    """A node created inside a UNIQUE sub-patch surfaces as an incremental on_node_added
    (root ≡ unique sub-patch), which wires its STATE_UPDATE forwarding exactly like a
    top-level add. Regression guard for the original bug: when this fired only
    on_subpatch_changed and forwarding was never wired, editing the member's param applied
    in the backend (its viewer updated) but the echo never reached the browser, so the
    param panel snapped back to the creation default."""
    manager = _bare_manager(use_multiprocessing=False)
    manager._layout = None
    try:
        hub = _hub(manager)
        manager._bridge = types.SimpleNamespace(control=hub)

        a = manager.add_node("Oscillator", "inputs")
        b = manager.add_node("Buffer", "signal")
        inst = manager.group_nodes([a, b])

        member = manager.add_member_node(inst, "Buffer", "signal")
        assert member in manager.nodes
        # The new member's NodeRef must carry its forwarding handlers so a later param
        # edit echoes back (state_update) instead of the panel reverting to default.
        assert MessageType.STATE_UPDATE in manager.nodes[member].callbacks, (
            "member status forwarding never wired"
        )
        assert member in hub._wired_nodes
    finally:
        manager.terminate(notify_gui=False)


def test_member_removal_is_incremental_with_membership():
    """Root ≡ unique sub-patch: removing a member NOT wired to any boundary is an
    incremental node_removed carrying the member's membership — the frontend drops the uid
    from the owning instance's members map (so member_count / slots stay correct) without a
    wholesale subpatch_changed snapshot. (A removal that UNWIRES a boundary IS structural
    and falls back to the snapshot — covered in test_root_scope.)"""
    manager = _bare_manager(use_multiprocessing=False)
    manager._layout = None
    try:
        hub = _hub(manager)
        manager._bridge = types.SimpleNamespace(control=hub)
        events: list = []
        hub.broadcast_threadsafe = lambda msg: events.append(msg)

        a = manager.add_node("Oscillator", "inputs")
        b = manager.add_node("Buffer", "signal")
        inst = manager.group_nodes([a, b])
        member = manager.add_member_node(inst, "Buffer", "signal")
        before = len(manager._instances[inst].members)

        events.clear()
        manager.remove_node(member)

        assert member not in manager.nodes
        assert len(manager._instances[inst].members) == before - 1
        kinds = [e["event"] for e in events]
        assert "node_removed" in kinds, "an unwired member removal is incremental"
        assert "subpatch_changed" not in kinds, "no snapshot needed for an unwired removal"
        payload = next(e["payload"] for e in events if e["event"] == "node_removed")
        assert payload["node"] == member
        assert payload["membership"]["instance"] == inst
    finally:
        manager.terminate(notify_gui=False)


def test_instantiate_definition_spawns_members_silently():
    """Members spawned by instantiate_definition (duplicate-as-shared) must surface ONLY
    via the trailing on_subpatch_changed snapshot — not a per-member on_node_added, which
    flashes them at root and leaves a phantom on the frontend if the wrapping transaction
    rolls back. (The snapshot's wiring sweep wires them, so silent spawn is safe.)"""
    manager = _bare_manager(use_multiprocessing=False)
    manager._layout = None
    try:
        hub = _hub(manager)
        manager._bridge = types.SimpleNamespace(control=hub)
        a = manager.add_node("Oscillator", "inputs")
        b = manager.add_node("Buffer", "signal")
        inst = manager.group_nodes([a, b])
        def_id = manager.share_instance(inst)

        events: list = []
        hub.broadcast_threadsafe = lambda msg: events.append(msg)
        new_inst = manager.instantiate_definition(def_id, (10, 10))

        kinds = [e["event"] for e in events]
        assert "subpatch_changed" in kinds
        assert "node_added" not in kinds, "members must spawn silently, surfaced by the snapshot"
        snap = next(e["payload"] for e in events if e["event"] == "subpatch_changed")
        assert new_inst in snap["instances"]
        assert snap["instances"][new_inst]["member_count"] == 2
    finally:
        manager.terminate(notify_gui=False)


def test_subpatch_changed_drops_wiring_for_vanished_member():
    """remove_instance tears members down with notify_gui=False (only on_subpatch_changed
    fires, no on_node_removed), so the wiring bookkeeping must be reconciled here too —
    else a uid lingers in _wired_nodes and a later same-uid reuse would early-return."""
    manager = _bare_manager(use_multiprocessing=False)
    manager._layout = None
    try:
        hub = _hub(manager)
        manager._bridge = types.SimpleNamespace(control=hub)

        a = manager.add_node("Oscillator", "inputs")
        b = manager.add_node("Buffer", "signal")
        inst = manager.group_nodes([a, b])
        assert a in hub._wired_nodes and b in hub._wired_nodes

        manager.remove_instance(inst)  # tears down a + b, fires on_subpatch_changed
        assert a not in manager.nodes and b not in manager.nodes
        assert a not in hub._wired_nodes and b not in hub._wired_nodes
    finally:
        manager.terminate(notify_gui=False)


def test_restart_node_op_respawns_member_in_place_preserving_membership():
    """The crash-recovery 'restart' must respawn a node IN PLACE (the manager's
    restart_node keeps uid + display name + sub-patch membership + links), not remove+add
    — which would land a sub-patch member back at ROOT and, for a SHARED member, cascade a
    mirror-remove across siblings. The bridge exposes restart_node so the frontend uses it."""
    manager = _bare_manager(use_multiprocessing=False)
    manager._layout = None
    try:
        hub = _hub(manager)

        async def _noop_rewire(_uid):
            return None

        # restart_node also schedules a data-plane mux re-point on the bridge loop;
        # give the fake bridge the same shape as the real server (control + data + schedule).
        manager._bridge = types.SimpleNamespace(
            control=hub,
            data=types.SimpleNamespace(rewire_node=_noop_rewire),
            schedule=lambda coro: coro.close(),
        )
        a = manager.add_node("Oscillator", "inputs")
        b = manager.add_node("Buffer", "signal")
        inst = manager.group_nodes([a, b])
        member = manager.add_member_node(inst, "Buffer", "signal")
        old_membership = dict(manager.nodes[member].membership)

        asyncio.run(hub._dispatch("restart_node", {"node": member}))

        assert member in manager.nodes  # uid stable, respawned in place
        assert manager.nodes[member].membership == old_membership  # stayed in its sub-patch
        assert manager._membership[member] == inst
    finally:
        manager.terminate(notify_gui=False)
