"""Grouping members that a boundary forwards into must RE-CHAIN the boundary through
the new nested instance — the chaining inverse of expand/remove's defensive unwire.

Before this fix `group_nodes` was the only structural mutator that re-homed members
WITHOUT repairing the holder scope's interface, so a boundary wired to a grouped member
was left dangling at a local that no longer existed in that scope. That is the root of
the reported cluster: "boundary <inst>:<bnd> inner member is gone", "re-pointing a
chained boundary isn't supported", and "grouping in front of an Out node orphans it
(no data out, can't reconnect)".
"""
from .test_manager import _bare_manager, _member


def _out_slot(mgr, uid):
    return list(mgr.nodes[uid].output_slots)[0]


def test_group_member_feeding_out_boundary_rechains_through_new_instance():
    """Inside a unique sub-patch S, member b0 feeds S's Out boundary. Grouping b0 into a
    fresh nested instance N must keep the Out boundary RESOLVABLE — chained S→N→b0 — not
    dangling. No external consumer here, so N gets a freshly-authored Out boundary."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        S = mgr.group_nodes([b0, b1])  # S::buffer0, S::buffer1
        out_slot = _out_slot(mgr, b0)
        bnd = mgr.add_boundary(S, "out", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer0", out_slot)
        assert mgr.resolve_boundary(S, bnd) == (b0, out_slot)

        N = mgr.group_nodes([b0])  # re-home b0 into a new nested instance inside S

        # The Out boundary still reaches b0's real leaf — now via the S→N→b0 chain.
        assert mgr.resolve_boundary(S, bnd) == (b0, out_slot)
        entry = mgr._instances[S].interface[bnd]
        # It now forwards THROUGH N (the nested instance), not at the gone direct member.
        assert entry.inner_node == mgr._instances[S].members[N]
        assert mgr._member_uid(S, entry.inner_node) == N
    finally:
        mgr.terminate()


def test_group_member_with_external_consumer_reuses_derived_boundary():
    """When an external consumer is wired to the Out port, grouping the producer creates
    a crossing flat link, so `_derive_interface` already authors N's Out boundary. The
    re-chain must REUSE it (not add a second exposing the same inner slot, which
    wire_boundary would reject)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        S = mgr.group_nodes([b0, b1])
        out_slot = _out_slot(mgr, b0)
        bnd = mgr.add_boundary(S, "out", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer0", out_slot)
        ext = mgr.add_node("Buffer", "signal")  # external consumer, outside S
        mgr.add_link(b0, ext, out_slot, "val")  # the spliced external link b0 → ext

        N = mgr.group_nodes([b0])  # crossing link b0→ext makes _derive_interface author N's Out

        assert mgr.resolve_boundary(S, bnd) == (b0, out_slot)
        # Exactly one Out boundary on N for b0 (reuse, not duplicate).
        outs = [e for e in mgr._instances[N].interface.values()
                if e.dir == "out" and e.inner_slot == out_slot]
        assert len(outs) == 1
    finally:
        mgr.terminate()


def test_group_member_fed_by_in_boundary_rechains_through_new_instance():
    """Symmetric to the Out case: an In boundary feeding a member's input must re-chain
    S→N→member when that member is grouped, so the input stays exposed."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        S = mgr.group_nodes([b0, b1])
        bnd = mgr.add_boundary(S, "in", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer0", "val")  # Buffer input slot
        assert mgr.resolve_boundary(S, bnd) == (b0, "val")

        N = mgr.group_nodes([b0])

        assert mgr.resolve_boundary(S, bnd) == (b0, "val")
        entry = mgr._instances[S].interface[bnd]
        assert mgr._member_uid(S, entry.inner_node) == N
    finally:
        mgr.terminate()
