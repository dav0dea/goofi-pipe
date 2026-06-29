"""Phase 4 — the bridge ships a COMPUTED, recursive per-instance snapshot record.

The frontend MIRRORS backend-computed state instead of re-deriving it. `describe_instance`
is the single server source of truth: it ships `parent` (the tree edge), `members`
inverted+enriched to {local: {uid, is_instance}}, server-resolved boundary dtypes (chain
to leaf), computed `slots`, `siblings`, `error` (first errored DESCENDANT across the whole
subtree), `member_count`, and `viewers`.
"""
import types

from goofi.bridge.control import ControlHub

from .test_manager import _bare_manager, _build_grouped_graph, _member


def _hub(manager):
    server = types.SimpleNamespace(manager=manager, schedule=lambda coro: coro.close())
    return ControlHub(server)


def test_snapshot_instance_record_is_computed_and_recursive():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        # inner > [oscillator0, select0]; inner OUT boundary wired to select0
        osc = mgr.add_node("Oscillator", "inputs")
        sel = mgr.add_node("Select", "array", params={"select": {"include": "0:5"}})
        mgr.add_link(osc, sel, "out", "data")
        inner = mgr.group_nodes([sel])
        sel_local = mgr._instances[inner].members[sel]
        out_slot = list(mgr.nodes[sel].output_slots)[0]
        inner_bnd = mgr.add_boundary(inner, "out", "ARRAY")
        mgr.wire_boundary(inner, inner_bnd, sel_local, out_slot)

        # outer > inner; outer OUT boundary chained to inner's boundary, plus one UNWIRED
        outer = mgr.group_nodes([inner])
        inner_local = mgr._instances[outer].members[inner]
        outer_bnd = mgr.add_boundary(outer, "out", "ARRAY")
        mgr.wire_boundary(outer, outer_bnd, inner_local, inner_bnd)
        unwired = mgr.add_boundary(outer, "in", "STRING")  # left unwired

        # a deliberately-errored deepest member
        mgr.nodes[sel].last_error = "boom in the deep"
        # persisted per-instance viewer state
        mgr._instances[outer].viewers = {"out0": {"kind": "line"}}

        snap = _hub(mgr)._snapshot()
        insts = snap["instances"]

        # --- parent (tree edge) ---
        assert insts[outer]["parent"] is None
        assert insts[inner]["parent"] == outer

        # --- members inverted + enriched ---
        m = insts[outer]["members"]
        assert inner_local in m and m[inner_local]["uid"] == inner
        assert m[inner_local]["is_instance"] is True
        assert insts[inner]["members"][sel_local]["is_instance"] is False
        assert insts[outer]["member_count"] == 1

        # --- interface dtype resolved chain-to-leaf; unwired keeps stored, no throw ---
        assert insts[outer]["interface"][outer_bnd]["dtype"] == "ARRAY"
        assert insts[outer]["interface"][unwired]["dtype"] == "STRING"

        # --- slots computed from WIRED boundaries only ---
        assert insts[outer]["slots"]["output"] == {outer_bnd: "ARRAY"}
        assert unwired not in insts[outer]["slots"]["input"]  # unwired -> not a slot

        # --- error = first errored DESCENDANT (subtree, not direct member) ---
        assert insts[outer]["error"] == "boom in the deep"
        assert insts[inner]["error"] == "boom in the deep"

        # --- viewers round-trip ---
        assert insts[outer]["viewers"] == {"out0": {"kind": "line"}}
    finally:
        mgr.terminate(notify_gui=False)


def test_snapshot_ships_shared_siblings():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)

        insts = _hub(mgr)._snapshot()["instances"]
        assert set(insts[inst]["siblings"]) == {inst2}
        assert set(insts[inst2]["siblings"]) == {inst}
        # a unique instance has no siblings
        uniq = mgr.group_nodes([mgr.add_node("Buffer", "signal")])
        insts = _hub(mgr)._snapshot()["instances"]
        assert insts[uniq]["siblings"] == []
    finally:
        mgr.terminate(notify_gui=False)


def test_instance_error_and_ancestors_for_live_error_propagation():
    """Phase 4 review #2: a member's live error must surface on its ancestor instances.
    The manager exposes the building blocks: _instance_error (first errored descendant)
    and _ancestor_instances (the chain a node-error event must refresh)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.add_link(a, b, "out", "val")
        inner = mgr.group_nodes([b])
        outer = mgr.group_nodes([inner])  # outer > inner > [b]
        assert mgr._instance_error(outer) is None
        assert mgr._ancestor_instances(b) == [inner, outer]

        mgr.nodes[b].last_error = "deep boom"
        assert mgr._instance_error(inner) == "deep boom"
        assert mgr._instance_error(outer) == "deep boom"  # surfaces up the whole chain
        mgr.nodes[b].last_error = None
        assert mgr._instance_error(outer) is None  # clears
    finally:
        mgr.terminate(notify_gui=False)


def test_wire_boundary_rejects_chaining_onto_unwired_nested_boundary():
    """Phase 4 review #8: chaining a boundary onto a nested instance's UNWIRED boundary
    would expose a port the collapsed node doesn't actually have. Reject it."""
    import pytest

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inner = _build_grouped_graph(mgr)
        inner_bnd = mgr.add_boundary(inner, "out", "ARRAY")  # added but NOT wired
        outer = mgr.group_nodes([inner])
        inner_local = mgr._instances[outer].members[inner]
        outer_bnd = mgr.add_boundary(outer, "out", "ARRAY")
        with pytest.raises(ValueError):
            mgr.wire_boundary(outer, outer_bnd, inner_local, inner_bnd)  # chain onto unwired child
    finally:
        mgr.terminate(notify_gui=False)
