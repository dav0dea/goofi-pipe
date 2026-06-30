"""Grouping members that a boundary forwards into must RE-CHAIN the boundary through
the new nested instance — the chaining inverse of expand/remove's defensive unwire.

Before this fix `group_nodes` was the only structural mutator that re-homed members
WITHOUT repairing the holder scope's interface, so a boundary wired to a grouped member
was left dangling at a local that no longer existed in that scope. That is the root of
the reported cluster: "boundary <inst>:<bnd> inner member is gone", "re-pointing a
chained boundary isn't supported", and "grouping in front of an Out node orphans it
(no data out, can't reconnect)".
"""
import pytest

from goofi.bridge.schemas import describe_instance

from .test_manager import _bare_manager, _build_grouped_graph, _member


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


def test_repoint_chained_out_boundary_to_direct_member_resplices_to_new_leaf():
    """A boundary chained through a nested instance can be re-pointed to a different inner
    target WITHOUT a manual unwire-first; the external link follows to the new leaf. This
    removes the old "re-pointing a chained boundary isn't supported" wall (symptom 1)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        S = mgr.group_nodes([b0, b1])
        out_slot = _out_slot(mgr, b0)
        bnd = mgr.add_boundary(S, "out", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer0", out_slot)
        ext = mgr.add_node("Buffer", "signal")
        mgr.add_link(b0, ext, out_slot, "val")  # external consumer of the Out port

        N = mgr.group_nodes([b0])  # S.bnd is now chained S→N→b0
        assert mgr._instances[S].interface[bnd].inner_node == mgr._instances[S].members[N]

        # Re-point the chained Out boundary to the still-direct member b1 — no unwire.
        b1_out = _out_slot(mgr, b1)
        mgr.wire_boundary(S, bnd, "buffer1", b1_out)

        assert mgr.resolve_boundary(S, bnd) == (b1, b1_out)
        # The external consumer follows to the new leaf: b1→ext now, b0→ext gone.
        assert any(l["node_out"] == b1 and l["node_in"] == ext for l in mgr.links)
        assert not any(l["node_out"] == b0 and l["node_in"] == ext for l in mgr.links)
    finally:
        mgr.terminate()


# --- Stage 3: a portal has its own renameable name (default in0/out0), never the
# connected node's name+slot (the "buffer0.out" complaint) ---------------------------


def test_authored_boundary_gets_default_in_out_name():
    """add_boundary names a fresh portal in0/out0/in1… — a default that follows node-style
    naming and is independent of whatever it later connects to."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([buf])
        b_in = mgr.add_boundary(inst, "in", "ARRAY")
        b_out = mgr.add_boundary(inst, "out", "ARRAY")
        b_in2 = mgr.add_boundary(inst, "in", "ARRAY")
        ifc = mgr._instances[inst].interface
        assert ifc[b_in].name == "in0"
        assert ifc[b_out].name == "out0"
        assert ifc[b_in2].name == "in1"
    finally:
        mgr.terminate()


def test_derived_boundary_name_is_clean_default_not_member_slot():
    """A boundary auto-derived by grouping must DISPLAY a clean in0/out0 name, not the
    connected member's name+slot (e.g. 'select0.data'). The internal interface key may
    stay member-derived, but the user-facing name never is."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # auto-derives one In boundary (osc→select0)
        ifc = mgr._instances[inst].interface
        assert ifc, "expected an auto-derived boundary"
        b = next(iter(ifc.values()))
        assert b.name == "in0"
        # The snapshot the frontend renders carries the clean name on each interface entry.
        desc = describe_instance(mgr, inst, mgr._instances[inst])
        bid = next(iter(desc["interface"]))
        assert desc["interface"][bid]["name"] == "in0"
    finally:
        mgr.terminate()


def test_boundary_name_survives_save_load(tmp_path):
    """The renameable name round-trips through the .gfi document."""
    fp = str(tmp_path / "named.gfi")
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([buf])
        bid = mgr.add_boundary(inst, "out", "ARRAY")
        assert mgr._instances[inst].interface[bid].name == "out0"
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        assert mgr2._instances[inst].interface[bid].name == "out0"
    finally:
        mgr2.terminate()


# --- Stage 4: rename a portal (its display + exposed-slot label) ----------------------


def test_rename_boundary_updates_name_keeps_routing_key():
    """Renaming a portal changes only its label — the routing key (bnd_id) is unchanged,
    so the data route / external wires keep resolving."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([buf])
        out_slot = _out_slot(mgr, buf)
        bid = mgr.add_boundary(inst, "out", "ARRAY")
        mgr.wire_boundary(inst, bid, "buffer0", out_slot)

        mgr.rename_boundary(inst, bid, "result")

        assert mgr._instances[inst].interface[bid].name == "result"
        assert mgr.resolve_boundary(inst, bid) == (buf, out_slot)  # key unchanged
    finally:
        mgr.terminate()


def test_rename_boundary_rejects_duplicate_same_dir_and_blank():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([buf])
        b0 = mgr.add_boundary(inst, "out", "ARRAY")  # out0
        b1 = mgr.add_boundary(inst, "out", "ARRAY")  # out1
        import pytest

        with pytest.raises(ValueError):
            mgr.rename_boundary(inst, b1, "out0")  # collides with b0
        with pytest.raises(ValueError):
            mgr.rename_boundary(inst, b1, "   ")  # blank
        # An in boundary may reuse an out name (separate slot namespaces).
        b_in = mgr.add_boundary(inst, "in", "ARRAY")
        mgr.rename_boundary(inst, b_in, "out0")
        assert mgr._instances[inst].interface[b_in].name == "out0"
    finally:
        mgr.terminate()


def test_rename_boundary_mirrors_across_shared_siblings():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([buf])
        bid = mgr.add_boundary(inst, "out", "ARRAY")
        def_id = mgr.share_instance(inst)
        sib = mgr.instantiate_definition(def_id)

        mgr.rename_boundary(inst, bid, "result")

        assert mgr._instances[sib].interface[bid].name == "result"
        assert mgr._definitions[def_id].interface[bid].name == "result"
    finally:
        mgr.terminate()


# --- Audit fixes: group<->expand symmetry, chained-IN guard, graceful re-chain ---------


def test_group_then_expand_round_trips_holder_boundary():
    """The inverse of the group re-chain (audit F4): expanding — or UNDOING — a group that
    re-chained a wired holder boundary must re-point it back to the lifted member, not tear
    it down. Otherwise undo of a group silently loses the boundary + its external link."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        S = mgr.group_nodes([b0, b1])
        out_slot = _out_slot(mgr, b0)
        bnd = mgr.add_boundary(S, "out", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer0", out_slot)
        ext = mgr.add_node("Buffer", "signal")
        mgr.add_link(b0, ext, out_slot, "val")  # external consumer of the Out port

        N = mgr.group_nodes([b0])  # re-chains S.bnd -> N -> b0
        assert mgr._instances[S].interface[bnd].inner_node == mgr._instances[S].members[N]

        mgr.expand_instance(N)  # the inverse: lift b0 back into S

        # The boundary forwards DIRECTLY at b0 again (round-trip), external link intact.
        assert mgr.resolve_boundary(S, bnd) == (b0, out_slot)
        assert any(l["node_out"] == b0 and l["node_in"] == ext for l in mgr.links)
    finally:
        mgr.terminate()


def test_group_internally_fed_input_unwires_boundary_instead_of_aborting():
    """Audit F3: an IN holder boundary onto a member input that becomes internally fed by
    another grouped member can't be chained — the group must still SUCCEED, tearing that
    boundary down, not roll the whole group back (a previously-allowed group)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        S = mgr.group_nodes([a, b])
        bnd = mgr.add_boundary(S, "in", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer0", "val")  # S.in -> a.val (unfed now)
        a_uid = _member(mgr, S, "buffer0")
        b_uid = _member(mgr, S, "buffer1")
        mgr.add_link(b_uid, a_uid, _out_slot(mgr, b_uid), "val")  # b -> a.val internal feed

        N = mgr.group_nodes([a_uid, b_uid])  # must NOT raise

        assert N in mgr._instances[S].members
        assert mgr._instances[S].interface[bnd].inner_node is None  # torn down gracefully
    finally:
        mgr.terminate()


def test_chained_in_repoint_rejects_evicting_an_internal_feed():
    """Audit F5: re-pointing an IN boundary onto a CHAINED target whose resolved leaf input
    is already fed by a link internal to the sub-patch must be REJECTED, not silently evict
    that internal feed via the single-source external splice."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        x = mgr.add_node("Buffer", "signal")
        l = mgr.add_node("Buffer", "signal")
        d = mgr.add_node("Buffer", "signal")
        p = mgr.add_node("Buffer", "signal")  # external producer
        mgr.add_link(x, l, _out_slot(mgr, x), "val")  # internal X -> L.val
        S = mgr.group_nodes([x, l, d])  # buffer0=x, buffer1=l, buffer2=d
        bnd = mgr.add_boundary(S, "in", "ARRAY")
        mgr.wire_boundary(S, bnd, "buffer2", "val")  # S.in -> d.val
        mgr.add_link(p, d, _out_slot(mgr, p), "val")  # external feed P -> d (the In port source)

        M = mgr.group_nodes([l])  # X->L crosses -> M auto-exposes L.val as an In boundary
        m_local = mgr._instances[S].members[M]
        m_in = next(k for k, e in mgr._instances[M].interface.items() if e.dir == "in")

        with pytest.raises(ValueError):
            mgr.wire_boundary(S, bnd, m_local, m_in)  # would evict X -> L.val

        # The internal feed survives (no silent eviction).
        assert any(li["node_out"] == x and li["node_in"] == l for li in mgr.links)
    finally:
        mgr.terminate()
