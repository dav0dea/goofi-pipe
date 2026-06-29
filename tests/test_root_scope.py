"""Root-as-scope unification: the root graph is a first-class SubPatchInstance
(parent=None) held in `_instances` under a reserved id, so a top-level node is a
member of ROOT exactly like a sub-patch member — one add path, one remove path, one
membership funnel, one event rule (root = the scope with no parent).

These tests pin the materialization (ROOT exists, is inert on the wire, every
top-level entity is its member) as it lands step by step.
"""
import types

from goofi.manager import ROOT_ID

from .test_control_ops import _hub
from .test_manager import _bare_manager, _build_grouped_graph, _member


def test_root_scope_is_materialized():
    """ROOT exists as a first-class instance the moment the manager is up: a unique,
    parent-less, boundary-less, def-less scope under the reserved id."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        assert ROOT_ID in mgr._instances
        root = mgr._instances[ROOT_ID]
        assert root.parent is None
        assert root.kind == "unique"
        assert root.def_id is None
        assert root.interface == {}  # the outermost scope has no boundaries
    finally:
        mgr.terminate(notify_gui=False)


def test_add_node_scope_is_the_one_entry_for_root_and_sub_patch():
    """One add path: add_node(scope=…) adds to ANY scope. Omitted/ROOT → a top-level
    node (member of ROOT); a sub-patch uid → a member of that instance, exactly like the
    old add_member_node (same local-keyed, shared-mirroring orchestration)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # default scope = ROOT
        assert mgr._membership[a] == ROOT_ID

        b = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a, b])

        # scope=inst lands the node inside the sub-patch (a member of the instance)
        member = mgr.add_node("Buffer", "signal", scope=inst)
        assert member in mgr._instances[inst].members
        assert mgr._membership[member] == inst
        # explicit ROOT scope == omitted scope
        c = mgr.add_node("Buffer", "signal", scope=ROOT_ID)
        assert mgr._membership[c] == ROOT_ID
    finally:
        mgr.terminate(notify_gui=False)


def test_root_dissolves_on_save():
    """Persistence stays runtime-only: ROOT never serializes as an instance — a flat
    graph saves with NO instances (ROOT is dissolved into the top-level nodes/links). The
    `.gfi` v2 format is unchanged by materializing ROOT in memory."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        mgr.add_node("Oscillator", "inputs")
        root_nodes, root_links, definitions, instances = mgr.build_v2_tree()
        assert ROOT_ID not in instances  # ROOT is dissolved, not emitted as an instance
        assert len(root_nodes) == 1  # the oscillator is a top-level node
        assert definitions == {}
    finally:
        mgr.terminate(notify_gui=False)


# --- Step 3: ROOT is exposed on the wire, root ≡ unique sub-patch end-to-end ----------


def test_root_ships_as_an_instance_in_the_snapshot():
    """Step 3 exposes ROOT on the live wire: the snapshot ships ROOT as just another
    instance (parent=None, no boundaries), and every top-level entity is its member — so
    the frontend's root scope is literally ROOT_ID, not a null special case."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        hub = _hub(mgr)
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a])  # a top-level instance; 'a' moves into it
        snap = hub._snapshot()
        assert ROOT_ID in snap["instances"]
        root = snap["instances"][ROOT_ID]
        assert root["parent"] is None  # ROOT is the top of the tree
        assert root["interface"] == {}  # the outermost scope has no boundaries
        member_uids = {m["uid"] for m in root["members"].values()}
        assert b in member_uids  # a top-level node is a ROOT member
        assert inst in member_uids  # a top-level instance is a ROOT member
        assert a not in member_uids  # 'a' moved into the sub-patch, no longer ROOT's
    finally:
        mgr.terminate(notify_gui=False)


def test_top_level_node_reports_root_membership_on_the_wire():
    """No more None shim: a top-level node reports its REAL membership
    ({instance: ROOT_ID, ...}), a member of ROOT exactly like a sub-patch member."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        hub = _hub(mgr)
        a = mgr.add_node("Oscillator", "inputs")
        node = next(n for n in hub._snapshot()["nodes"] if n["uid"] == a)
        assert node["membership"] is not None
        assert node["membership"]["instance"] == ROOT_ID
    finally:
        mgr.terminate(notify_gui=False)


def test_top_level_instance_reports_root_parent_on_the_wire():
    """No more None shim: a top-level instance reports parent=ROOT_ID."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        hub = _hub(mgr)
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        assert hub._snapshot()["instances"][inst]["parent"] == ROOT_ID
    finally:
        mgr.terminate(notify_gui=False)


def test_add_member_to_unique_subpatch_is_incremental_node_added():
    """Root ≡ unique sub-patch: adding a member to a UNIQUE sub-patch is an incremental
    node_added carrying the member's real membership — exactly like adding a top-level
    node — NOT a wholesale subpatch_changed snapshot."""
    mgr = _bare_manager(use_multiprocessing=False)
    mgr._layout = None
    try:
        hub = _hub(mgr)
        mgr._bridge = types.SimpleNamespace(control=hub)
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a, b])

        events: list = []
        hub.broadcast_threadsafe = lambda msg: events.append(msg)
        member = mgr.add_member_node(inst, "Buffer", "signal")

        kinds = [e["event"] for e in events]
        assert "node_added" in kinds
        assert "subpatch_changed" not in kinds
        payload = next(e["payload"] for e in events if e["event"] == "node_added")
        assert payload["uid"] == member
        assert payload["membership"]["instance"] == inst
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_unwired_member_is_incremental_node_removed_with_membership():
    """Removing a member NOT wired to any boundary is an incremental node_removed whose
    payload carries the member's membership (so the frontend drops it from the owning
    instance's members map) — no structural snapshot needed."""
    mgr = _bare_manager(use_multiprocessing=False)
    mgr._layout = None
    try:
        hub = _hub(mgr)
        mgr._bridge = types.SimpleNamespace(control=hub)
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a, b])
        member = mgr.add_member_node(inst, "Buffer", "signal")

        events: list = []
        hub.broadcast_threadsafe = lambda msg: events.append(msg)
        mgr.remove_node(member)

        kinds = [e["event"] for e in events]
        assert "node_removed" in kinds
        assert "subpatch_changed" not in kinds
        payload = next(e["payload"] for e in events if e["event"] == "node_removed")
        assert payload["node"] == member
        assert payload["membership"]["instance"] == inst
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_wired_member_is_structural_subpatch_changed():
    """Removing a member that IS wired to a boundary unwires that boundary — the
    instance's computed slots/interface change — so it fires a full subpatch_changed
    snapshot, not a bare node_removed that would leave the instance record stale."""
    mgr = _bare_manager(use_multiprocessing=False)
    mgr._layout = None
    try:
        hub = _hub(mgr)
        mgr._bridge = types.SimpleNamespace(control=hub)
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        local = mgr._instances[inst].members[a]
        out_slot = list(mgr.nodes[a].output_slots)[0]
        dtype = mgr.nodes[a].output_slots[out_slot].name
        bnd = mgr.add_boundary(inst, "out", dtype)
        mgr.wire_boundary(inst, bnd, local, out_slot)

        events: list = []
        hub.broadcast_threadsafe = lambda msg: events.append(msg)
        mgr.remove_node(a)

        kinds = [e["event"] for e in events]
        assert "subpatch_changed" in kinds
        assert "node_removed" not in kinds
    finally:
        mgr.terminate(notify_gui=False)


def test_structural_ops_reject_the_root_scope():
    """ROOT is a scope but NOT a sub-patch — it has no parent, no boundaries, and can't be
    dissolved/shared/expanded. Now that ROOT ships on the wire (so the bridge's remove_node
    op could route ROOT_ID to remove_instance), the structural mutators reject it explicitly
    rather than corrupting the canvas (the editor never targets ROOT, but the backend is the
    authority)."""
    import pytest

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        with pytest.raises(ValueError):
            mgr.remove_instance(ROOT_ID)
        with pytest.raises(ValueError):
            mgr.expand_instance(ROOT_ID)
        with pytest.raises(ValueError):
            mgr.share_instance(ROOT_ID)
        with pytest.raises(ValueError):
            mgr.make_unique(ROOT_ID)
        with pytest.raises(ValueError):
            mgr.add_boundary(ROOT_ID, "in", "ARRAY")
        # ROOT survives every rejected op, still the parent-less root scope.
        assert ROOT_ID in mgr._instances
        assert mgr._instances[ROOT_ID].parent is None
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_top_level_node_carries_root_membership():
    """A plain top-level remove is the same incremental node_removed — its payload carries
    {instance: ROOT_ID} so the frontend drops it from ROOT's members map (the index the
    canvas renders root children from)."""
    mgr = _bare_manager(use_multiprocessing=False)
    mgr._layout = None
    try:
        hub = _hub(mgr)
        mgr._bridge = types.SimpleNamespace(control=hub)
        a = mgr.add_node("Oscillator", "inputs")

        events: list = []
        hub.broadcast_threadsafe = lambda msg: events.append(msg)
        mgr.remove_node(a)

        payload = next(e["payload"] for e in events if e["event"] == "node_removed")
        assert payload["node"] == a
        assert payload["membership"]["instance"] == ROOT_ID
    finally:
        mgr.terminate(notify_gui=False)
