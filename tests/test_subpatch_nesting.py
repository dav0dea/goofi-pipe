"""Phase 3a — truly recursive sub-patch MEMBERSHIP (a tree, not one level).

The live execution graph stays flat (real node processes); only the sub-patch
STRUCTURE becomes a tree. A sub-patch instance is a first-class, uid-keyed entity
that can itself be a member of another instance. The parent edge is carried in two
lockstep views — `_membership` (the unified upward index, keyed by ANY entity uid:
node OR instance) and `SubPatchInstance.parent` (the instance-side marker) — kept
in sync by the one `_attach_member`/`_detach_member` funnel, exactly like a node's
`_membership[uid]` + `ref.membership` marker.
"""
from .test_manager import _bare_manager, _build_grouped_graph, _member
from .test_subpatch_invariants import assert_subpatch_invariants


def _second_instance(mgr):
    """A second top-level instance B = [buffer0, buffer1]; returns its uid."""
    n0 = mgr.add_node("Buffer", "signal")
    n1 = mgr.add_node("Buffer", "signal")
    return mgr.group_nodes([n0, n1])


def test_attach_detach_tracks_instance_parent():
    """The membership funnel keeps an instance member's `.parent` marker in lockstep
    with the `_membership` upward index — the instance-side analog of a node's
    `ref.membership` marker."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, A = _build_grouped_graph(mgr)  # instance A: [select0, select1]
        B = _second_instance(mgr)  # a second top-level instance
        assert mgr._instances[B].parent is None

        mgr._attach_member(A, B, "subpatch1")  # nest B under A via the funnel
        assert mgr._instances[B].parent == A  # instance-side marker
        assert mgr._membership[B] == A  # upward index keyed by the instance uid
        assert B in mgr._instances[A].members  # forward index

        mgr._detach_member(B)
        assert mgr._instances[B].parent is None
        assert B not in mgr._membership
    finally:
        mgr.terminate(notify_gui=False)


def test_group_node_and_instance_into_outer():
    """An existing sub-patch instance can itself be grouped as a member of an outer
    instance; the parent<->child edge is recorded in lockstep across the upward index
    and the instance-side marker, and the inner subtree is untouched."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inner = _build_grouped_graph(mgr)  # inner: [select0, select1]
        buf = mgr.add_node("Buffer", "signal")  # a top-level sibling node

        outer = mgr.group_nodes([buf, inner])  # group a node + an EXISTING instance

        assert inner in mgr._instances[outer].members  # instance-as-member
        assert buf in mgr._instances[outer].members
        assert mgr._membership[inner] == outer  # upward index keyed by instance uid
        assert mgr._membership[buf] == outer
        assert mgr._instances[inner].parent == outer  # instance-side marker, lockstep
        assert mgr._instances[outer].parent is None  # outer is top-level

        s0 = _member(mgr, inner, "select0")
        assert mgr._membership[s0] == inner  # inner's subtree still belongs to inner
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_group_subset_of_instance_nests_child():
    """Grouping a SUBSET of an existing instance's members re-homes them one level
    down into a new child nested under that instance — the members leave the parent's
    `members`, the child joins it, and every index stays consistent."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        c = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b, c])

        child = mgr.group_nodes([a, b])  # group a subset of P's members

        assert mgr._membership[a] == child  # a, b re-homed into the child
        assert mgr._membership[b] == child
        assert mgr._instances[child].parent == P  # child nested under P
        assert mgr._membership[child] == P
        assert child in mgr._instances[P].members  # P now owns the child
        assert c in mgr._instances[P].members  # c still directly in P
        assert a not in mgr._instances[P].members  # a moved down out of P
        assert b not in mgr._instances[P].members
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_group_across_scopes_is_rejected():
    """Members at different nesting levels have no single place to land — reject, and
    leave the graph untouched (the check runs before any mutation)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b])  # a, b now inside P
        x = mgr.add_node("Buffer", "signal")  # a top-level node

        import pytest

        with pytest.raises(ValueError):
            mgr.group_nodes([a, x])  # a is inside P, x is top-level
        assert set(mgr._instances[P].members) == {a, b}  # unchanged
        assert mgr._membership.get(x) is None
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_group_inside_shared_parent_is_rejected():
    """Nesting a child inside a SHARED instance would have to mirror the new structure
    into the definition + every sibling family — deferred to Phase 3d. Until then it is
    rejected cleanly, leaving the definition and members untouched."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        c = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b, c])
        def_id = mgr.share_instance(P)
        before_def = set(mgr._definitions[def_id].members)

        import pytest

        with pytest.raises(ValueError):
            mgr.group_nodes([a, b])  # nesting inside a shared instance
        assert set(mgr._definitions[def_id].members) == before_def  # def untouched
        assert set(mgr._instances[P].members) == {a, b, c}  # members untouched
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_expand_child_lifts_members_into_parent():
    """Expanding a nested child dissolves it INTO its parent (not to top-level): the
    child's members become direct members of the parent, the child record is gone."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        c = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b, c])
        child = mgr.group_nodes([a, b])  # P > child > [a, b];  c direct in P

        mgr.expand_instance(child)

        assert mgr._membership[a] == P  # lifted INTO P, not to top-level
        assert mgr._membership[b] == P
        assert a in mgr._instances[P].members
        assert b in mgr._instances[P].members
        assert c in mgr._instances[P].members
        assert child not in mgr._instances  # child record dissolved
        assert child not in mgr._instances[P].members
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_expand_middle_reparents_nested_instance():
    """Expanding a MIDDLE instance lifts its members up one level — including a nested
    INSTANCE member, whose whole subtree is reparented (not flattened)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        g0 = mgr.add_node("Buffer", "signal")
        g1 = mgr.add_node("Buffer", "signal")
        G = mgr.group_nodes([g0, g1])  # innermost instance
        extra = mgr.add_node("Buffer", "signal")
        C = mgr.group_nodes([G, extra])  # C > [G, extra]
        P = mgr.group_nodes([C])  # P > C > G

        assert mgr._instances[G].parent == C  # precondition: G nested in C

        mgr.expand_instance(C)  # dissolve the middle level

        assert mgr._instances[G].parent == P  # G's subtree reparented up to P
        assert G in mgr._instances[P].members
        assert G in mgr._instances  # G itself preserved (NOT flattened)
        assert mgr._membership[g0] == G  # G's own members untouched
        assert C not in mgr._instances  # middle level dissolved
        assert mgr._membership[extra] == P  # C's node member lifted into P too
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_outer_deletes_nested_subtree():
    """Deleting an instance recursively removes its whole subtree — nested instances
    and their member nodes — leaving every index clean (and untouched siblings live)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inner = _build_grouped_graph(mgr)  # inner: [select0, select1]
        s0 = _member(mgr, inner, "select0")
        s1 = _member(mgr, inner, "select1")
        buf = mgr.add_node("Buffer", "signal")
        outer = mgr.group_nodes([buf, inner])  # outer > [buf, inner > [s0, s1]]

        mgr.remove_instance(outer)

        assert outer not in mgr._instances
        assert inner not in mgr._instances  # nested instance removed too
        assert inner not in mgr._membership
        assert buf not in mgr.nodes  # direct node member removed
        assert s0 not in mgr.nodes and s1 not in mgr.nodes  # grandchild nodes removed
        assert osc in mgr.nodes  # an untouched outside node survives
        assert_subpatch_invariants(mgr)
    finally:
        mgr.terminate(notify_gui=False)
