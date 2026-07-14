"""Regression tests for the Phase 3 adversarial-review findings.

The recursive sub-patch structure (3a-3d) shipped with a class of defects around
operations that mutate STRUCTURE inside a shared family, plus a few boundary/
persistence/rollback edge cases. Each test here pins one confirmed finding.

Design principle these lock in: structural edits to a member of a SHARED sub-patch
are not supported — make the parent unique first (symmetric to group_nodes' existing
'cannot group inside a shared sub-patch' guard and remove_node's shared-member guard).
"""
import pytest

from goofi.manager import ROOT_ID

from .test_manager import _bare_manager, _member
from .test_subpatch_invariants import assert_subpatch_invariants


def _shared_outer_with_child(mgr):
    """outer (shared, def_outer) > child (shared, def_child) > [oscillator0, buffer0].
    Returns (outer, child, def_outer, def_child)."""
    a = mgr.add_node("Oscillator", "inputs")
    b = mgr.add_node("Buffer", "signal")
    mgr.add_link(a, b, "out", "val")
    child = mgr.group_nodes([a, b])
    outer = mgr.group_nodes([child])
    def_outer = mgr.share_instance(outer)
    def_child = mgr._instances[child].def_id
    return outer, child, def_outer, def_child


# === Cluster A: structural edits on a member of a SHARED parent are rejected ===

def test_expand_member_of_shared_parent_is_rejected():
    """Findings #1, #7: expanding a child whose parent is shared would lift node members
    straight into the shared parent without mirroring to its def/siblings (corruption)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        outer, child, def_outer, def_child = _shared_outer_with_child(mgr)
        with pytest.raises(ValueError):
            mgr.expand_instance(child)
        # nothing changed: families + defs intact
        assert mgr._instances[child].def_id == def_child
        assert mgr._instances[outer].def_id == def_outer
        assert def_child in mgr._definitions and def_outer in mgr._definitions
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_member_of_shared_parent_is_rejected():
    """Finding #6: deleting a nested member of a shared parent would corrupt the family
    and tear down node processes out from under the definition."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        outer, child, def_outer, def_child = _shared_outer_with_child(mgr)
        outer2 = mgr.instantiate_definition(def_outer)
        before_nodes = set(mgr.nodes)
        with pytest.raises(ValueError):
            mgr.remove_instance(child)
        assert set(mgr.nodes) == before_nodes  # no node torn down
        assert mgr._instances[child].def_id == def_child
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_make_unique_member_of_shared_parent_is_rejected():
    """Findings #5, #8: privatizing a single nested child of a shared parent would
    orphan the child def the parent def still references (def->def dangling)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        outer, child, def_outer, def_child = _shared_outer_with_child(mgr)
        with pytest.raises(ValueError):
            mgr.make_unique(child)
        # def_child still referenced by def_outer.instances and still has a live instance
        assert def_child in mgr._definitions
        assert any(ref["def"] == def_child for ref in mgr._definitions[def_outer].instances.values())
        assert mgr._instances[child].def_id == def_child
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_make_unique_outer_still_privatizes_whole_subtree():
    """The guard must NOT break the legitimate top-down privatization: make_unique on the
    ROOT shared parent still recursively privatizes its nested child (findings #5/#8 fix
    must keep the 3d recursive make_unique working)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        outer, child, def_outer, def_child = _shared_outer_with_child(mgr)
        mgr.make_unique(outer)  # outer.parent is None -> allowed
        assert mgr._instances[outer].def_id is None
        assert mgr._instances[child].def_id is None  # subtree privatized
        # sole-instance defs GC'd together (no dangling def->def reference)
        assert def_outer not in mgr._definitions
        assert def_child not in mgr._definitions
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_outer_still_removes_whole_shared_subtree():
    """The guard must NOT break removing the root: remove_instance on the root shared
    parent still recursively deletes its nested subtree."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        outer, child, def_outer, def_child = _shared_outer_with_child(mgr)
        leaves = set(mgr._instances[child].members)
        mgr.remove_instance(outer)  # root -> allowed
        assert outer not in mgr._instances and child not in mgr._instances
        assert leaves.isdisjoint(set(mgr.nodes))  # leaf nodes gone
        assert def_outer not in mgr._definitions and def_child not in mgr._definitions
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


# === Cluster B/C/D: expand local-dedup, defensive boundary unwire, subtree links ===

def test_expand_dedups_lifted_member_local_against_parent():
    """Finding #2: a lifted member's local must be deduped against the receiving parent
    (member locals are decoupled from reusable display names), or two members collide."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        c = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b, c])
        C = mgr.group_nodes([b, c])  # C nested under P; P keeps a@local 'buffer0'
        mgr.rename_node(a, "renamed")  # frees display 'buffer0' but P.members[a] local stays 'buffer0'
        d = mgr.add_member_node(C, "Buffer", "signal")  # d gets display + local 'buffer0' in C

        mgr.expand_instance(C)  # lift b, c, d into P; d's display 'buffer0' must not clash a's local

        locals_ = list(mgr._instances[P].members.values())
        assert len(locals_) == len(set(locals_)), f"duplicate local after expand: {locals_}"
        # every member is still uniquely resolvable
        for local in locals_:
            assert mgr._member_uid(P, local) is not None
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_expand_unwires_parent_boundary_that_forwarded_into_it():
    """Findings #3, #10: when an instance a parent boundary forwards into is dissolved,
    the parent boundary must not be left dangling at a gone member."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        # inner > [select0, select1]; inner OUT boundary -> select1
        from .test_manager import _build_grouped_graph
        osc, inner = _build_grouped_graph(mgr)
        s1 = _member(mgr, inner, "select1")
        out_slot = list(mgr.nodes[s1].output_slots)[0]
        inner_bnd = mgr.add_boundary(inner, "out", "ARRAY")
        mgr.wire_boundary(inner, inner_bnd, "select1", out_slot)
        outer = mgr.group_nodes([inner])
        inner_local = mgr._instances[outer].members[inner]
        outer_bnd = mgr.add_boundary(outer, "out", "ARRAY")
        mgr.wire_boundary(outer, outer_bnd, inner_local, inner_bnd)  # outer fwd -> inner's boundary

        mgr.expand_instance(inner)  # dissolves inner; outer_bnd's target is gone

        # outer_bnd must not dangle at the gone inner local
        e = mgr._instances[outer].interface[outer_bnd]
        assert e.inner_node is None or e.inner_node in mgr._instances[outer].members.values()
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_unwire_chained_boundary_preserves_internal_link_in_nested_child():
    """Finding #4: unwiring a chained outer boundary must not tear down a link that is
    INTERNAL to the nested child (the externality test must be subtree-aware)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        prod = mgr.add_node("Oscillator", "inputs")
        cons = mgr.add_node("Buffer", "signal")
        mgr.add_link(prod, cons, "out", "val")  # internal link inside `inner`
        inner = mgr.group_nodes([prod, cons])
        out_slot = list(mgr.nodes[prod].output_slots)[0]
        inner_bnd = mgr.add_boundary(inner, "out", "ARRAY")
        mgr.wire_boundary(inner, inner_bnd, mgr._instances[inner].members[prod], out_slot)
        outer = mgr.group_nodes([inner])
        inner_local = mgr._instances[outer].members[inner]
        outer_bnd = mgr.add_boundary(outer, "out", "ARRAY")
        mgr.wire_boundary(outer, outer_bnd, inner_local, inner_bnd)

        # an EXTERNAL consumer of the chained port
        ext = mgr.add_node("Buffer", "signal")
        leaf, lslot = mgr.resolve_boundary(outer, outer_bnd)
        mgr.add_link(leaf, ext, lslot, "val")

        mgr.wire_boundary(outer, outer_bnd, None, None)  # unwire the chained boundary

        # the EXTERNAL link is gone, but the INTERNAL prod->cons link survives
        assert not any(l["node_out"] == leaf and l["node_in"] == ext for l in mgr.links)
        assert any(l["node_out"] == prod and l["node_in"] == cons for l in mgr.links), (
            "internal nested-child link was wrongly torn down"
        )
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


# === Cluster E: _transaction rollback restores a node's owning scope =============

def test_transaction_rollback_restores_node_membership_marker():
    """Findings #9, #12: a rolled-back transaction must leave a surviving node's owning
    scope unchanged. The membership marker is derived from the _membership / members maps
    (both snapshotted + restored by _transaction), so the derived marker must be back to
    its pre-block value — a desync the invariant checker catches."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        c = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b, c])
        marker_a = dict(mgr._membership_marker(a))  # {'instance': P, 'local_name': ...}

        # group a subset into a new child, but force a failure mid-transaction
        real_attach = mgr._attach_member
        calls = {"n": 0}

        def boom(inst_id, uid, local):
            calls["n"] += 1
            real_attach(inst_id, uid, local)
            if calls["n"] == 2:  # after a's and b's markers were repointed to the child
                raise RuntimeError("forced failure")

        import pytest
        from unittest.mock import patch
        with patch.object(mgr, "_attach_member", boom):
            with pytest.raises(RuntimeError):
                mgr.group_nodes([a, b])

        # a's marker must be restored to P (not left pointing at the rolled-back child)
        assert mgr._membership_marker(a) == marker_a
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


# === #13: load precheck validates a definition's nested def->def references =====

def test_expand_doc_precheck_catches_dangling_def_to_def_reference():
    """Finding #13: the load precheck validated per-instance shared refs but not a
    definition's own nested `instances` (def->child_def) refs, deferring a corrupt-doc
    failure past the fail-fast precheck."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        root_nodes = {}
        root_links = []
        # a definition that references a child def which is absent everywhere
        definitions = {
            "defP": {"members": {}, "links": [], "interface": {},
                     "instances": {"child0": {"def": "defGHOST", "pos": [0, 0]}}},
        }
        instances = {
            "instP": {"kind": "shared", "def": "defP", "pos": [0, 0], "members": {}, "name": "subpatch0"},
        }
        with pytest.raises(KeyError):
            mgr._expand_doc(root_nodes, root_links, instances, definitions)
        # fail-fast: nothing spliced (only the ever-present ROOT scope remains)
        assert set(mgr._instances) == {ROOT_ID}
    finally:
        mgr.terminate(notify_gui=False)
