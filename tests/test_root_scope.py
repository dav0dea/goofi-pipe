"""Root-as-scope unification: the root graph is a first-class SubPatchInstance
(parent=None) held in `_instances` under a reserved id, so a top-level node is a
member of ROOT exactly like a sub-patch member — one add path, one remove path, one
membership funnel, one event rule (root = the scope with no parent).

These tests pin the materialization (ROOT exists, is inert on the wire, every
top-level entity is its member) as it lands step by step.
"""
from goofi.manager import ROOT_ID

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


def test_root_is_inert_on_the_wire():
    """Step 1 keeps the protocol identical: ROOT never appears as a sub-patch instance
    in the snapshot, and a flat graph still serializes with NO instances (ROOT is
    dissolved into the top-level nodes/links at save)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        mgr.add_node("Oscillator", "inputs")
        root_nodes, root_links, definitions, instances = mgr.build_v2_tree()
        assert ROOT_ID not in instances  # ROOT is dissolved, not emitted as an instance
        assert len(root_nodes) == 1  # the oscillator is a top-level node
        assert definitions == {}
    finally:
        mgr.terminate(notify_gui=False)
