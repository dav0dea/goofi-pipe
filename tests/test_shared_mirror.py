"""Strict-mirror surfaces sibling failures instead of swallowing them (backlog #8).

Editing a shared sub-patch member mirrors the edit to every sibling instance. A
sibling that fails to apply the edit must no longer be silently swallowed (which let
a shared family drift apart unnoticed) — the failure is surfaced (logged + pushed to
the UI as an error event). We assert the bridge error event, which is robust to global
logging state (unlike caplog under the full suite).
"""
from goofi.manager import ROOT_ID

from .test_manager import _bare_manager


class _Ctrl:
    def __init__(self):
        self.errors = []

    def broadcast_threadsafe(self, payload):
        self.errors.append(payload)

    def __getattr__(self, _name):
        # Tolerate any other control callback (on_subpatch_changed, on_link_added, …)
        # as a no-op so a fake bridge can be attached for just the surfacing assertion.
        return lambda *a, **k: None


class _FakeBridge:
    def __init__(self):
        self.control = _Ctrl()


def test_set_expression_mirrors_and_persists_across_shared_siblings():
    # Strict mirror must cover EXPRESSION edits, not just value edits: binding an
    # expression on one shared member must reach every sibling AND the definition
    # (so a freshly-instantiated sibling inherits it).
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)
        m1 = next(iter(mgr._instances[inst1].members))
        m2 = next(iter(mgr._instances[inst2].members))

        mgr.set_expression(m1, "oscillator", "frequency", "5 + 5", enabled=True)

        # the existing sibling mirrors the binding...
        p2 = mgr.nodes[m2].params["oscillator"]["frequency"]
        assert p2.expression == "5 + 5"
        assert p2.expression_enabled is True

        # ...the definition persists it...
        local = mgr._instances[inst1].members[m1]
        defrec = mgr._definitions[def_id].members[local]
        assert defrec["params"]["oscillator"]["frequency"]["expression"] == "5 + 5"

        # ...so a fresh sibling inherits it too.
        inst3 = mgr.instantiate_definition(def_id)
        m3 = next(iter(mgr._instances[inst3].members))
        assert mgr.nodes[m3].params["oscillator"]["frequency"].expression == "5 + 5"
    finally:
        mgr.terminate(notify_gui=False)


def test_value_edit_preserves_a_shared_members_stashed_expression_in_the_def():
    # A param can carry a STASHED expression (fx toggled off) while its value
    # widget stays editable. Dragging the value must not wipe the stashed
    # expression from the definition (the save source of truth).
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        m1 = next(iter(mgr._instances[inst1].members))
        local = mgr._instances[inst1].members[m1]

        mgr.set_expression(m1, "oscillator", "frequency", "3 + 4", enabled=False)
        mgr.update_param(m1, "oscillator", "frequency", 9.0)

        defrec = mgr._definitions[def_id].members[local]["params"]["oscillator"]["frequency"]
        assert isinstance(defrec, dict), "value edit clobbered the stashed expression dict"
        assert defrec["expression"] == "3 + 4"
        assert defrec["value"] == 9.0
    finally:
        mgr.terminate(notify_gui=False)


def test_internal_member_link_mirrors_to_siblings_and_definition():
    # Wiring two members of a SHARED sub-patch is a topology edit that must mirror:
    # into the definition (so it persists + new siblings inherit it) and into every
    # existing sibling's corresponding members.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst1 = mgr.group_nodes([a, b])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)
        la = mgr._instances[inst1].members[a]
        lb = mgr._instances[inst1].members[b]

        mgr.add_link(a, b, "out", "val")

        # definition carries the local-form link
        assert {"node_out": la, "node_in": lb, "slot_out": "out", "slot_in": "val"} in mgr._definitions[def_id].links
        # the existing sibling got the corresponding live link
        sa, sb = mgr._member_uid(inst2, la), mgr._member_uid(inst2, lb)
        assert {"node_out": sa, "node_in": sb, "slot_out": "out", "slot_in": "val"} in mgr._links
        # a fresh sibling inherits it
        inst3 = mgr.instantiate_definition(def_id)
        ta, tb = mgr._member_uid(inst3, la), mgr._member_uid(inst3, lb)
        assert {"node_out": ta, "node_in": tb, "slot_out": "out", "slot_in": "val"} in mgr._links

        # removing it mirrors too (def + siblings)
        mgr.remove_link(a, b, "out", "val")
        assert {"node_out": la, "node_in": lb, "slot_out": "out", "slot_in": "val"} not in mgr._definitions[def_id].links
        assert {"node_out": sa, "node_in": sb, "slot_out": "out", "slot_in": "val"} not in mgr._links
    finally:
        mgr.terminate(notify_gui=False)


def test_member_nd_cross_ref_mirrors_flat_per_instance(tmp_path):
    # An intra-sub-patch nd() cross-reference (one member references a fellow by its
    # FLAT name) must point at EACH instance's OWN member — on the existing sibling
    # (mirror), a freshly instantiated sibling, and after save/load. The definition
    # stores the ref in template-local form; each instance re-points it at its members.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a, b])
        la, lb = mgr._instances[inst1].members[a], mgr._instances[inst1].members[b]
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)  # existing sibling

        # bind a cross-ref on inst1's member b -> a, by a's flat display name
        mgr.set_expression(b, "oscillator", "frequency", f"nd('{mgr.nodes[a].name}')", enabled=True)

        def expr_of(iid):
            return mgr.nodes[mgr._member_uid(iid, lb)].params["oscillator"]["frequency"].expression

        def a_name_of(iid):
            return mgr.nodes[mgr._member_uid(iid, la)].name

        # the existing sibling's copy references ITS OWN 'a' member by that flat name...
        assert expr_of(inst2) == f"nd('{a_name_of(inst2)}')"
        # ...and so does a freshly instantiated sibling.
        inst3 = mgr.instantiate_definition(def_id)
        assert expr_of(inst3) == f"nd('{a_name_of(inst3)}')"

        fp = str(tmp_path / "req.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate(notify_gui=False)

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        for iid in [i for i in mgr2._instances if i != ROOT_ID]:
            b_u, a_u = mgr2._member_uid(iid, lb), mgr2._member_uid(iid, la)
            assert mgr2.nodes[b_u].params["oscillator"]["frequency"].expression == f"nd('{mgr2.nodes[a_u].name}')"
    finally:
        mgr2.terminate(notify_gui=False)


def test_shared_mirror_surfaces_sibling_failure():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        member1 = next(iter(mgr._instances[inst1].members))
        member2 = next(iter(mgr._instances[inst2].members))

        # make the sibling's mirror update raise
        def _boom(*_a, **_k):
            raise RuntimeError("sibling unreachable")

        mgr.nodes[member2].update_param = _boom

        # attach a bridge only for the edit, so its mirror failure is reported to the UI
        fake = _FakeBridge()
        mgr._bridge = fake
        mgr.update_param(member1, "oscillator", "frequency", 5.0)

        assert any(
            e.get("payload", {}).get("node") == member2 and "mirror" in e["payload"]["error"].lower()
            for e in fake.control.errors
        ), f"sibling mirror failure not surfaced to the UI; got: {fake.control.errors}"
    finally:
        mgr._bridge = None
        mgr.terminate(notify_gui=False)


def test_boundary_mirror_surfaces_sibling_failure_without_aborting():
    """A boundary edit on a shared instance mirrors to every sibling. If one sibling's
    interface can't be written (e.g. it was concurrently removed), the failure must be
    SURFACED (logged + UI error) — not propagated to abort the whole op, nor silently
    swallowed — so the edited instance + def + reachable siblings still update. Same
    'surface, don't swallow' contract the param/expression mirror already honours."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        class _BoomDict(dict):
            def __setitem__(self, k, v):
                raise RuntimeError("sibling interface unreachable")

        mgr._instances[inst2].interface = _BoomDict(mgr._instances[inst2].interface)
        fake = _FakeBridge()
        mgr._bridge = fake

        bnd = mgr.add_boundary(inst1, "out", "array")  # must NOT raise

        # edited instance + definition still got the boundary
        assert bnd in mgr._instances[inst1].interface
        assert bnd in mgr._definitions[def_id].interface
        # the sibling failure was surfaced to the UI
        assert any(
            e.get("payload", {}).get("node") == inst2 and "mirror" in e["payload"]["error"].lower()
            for e in fake.control.errors
        ), f"boundary mirror failure not surfaced; got: {fake.control.errors}"
    finally:
        mgr._bridge = None
        mgr.terminate(notify_gui=False)


def test_internal_link_mirror_surfaces_sibling_failure_without_aborting():
    """Wiring two members of a shared sub-patch mirrors the link to siblings. If a
    sibling's wire fails, surface it (don't abort the edit nor swallow it): the edited
    instance + def still carry the link, and the failure reaches the UI."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst1 = mgr.group_nodes([a, b])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)
        la, lb = mgr._instances[inst1].members[a], mgr._instances[inst1].members[b]
        sb = mgr._member_uid(inst2, lb)

        def _boom(*_a, **_k):
            raise RuntimeError("sibling wire unreachable")

        mgr.nodes[sb].subscribe_input = _boom  # the sibling's wire fails mid-fan-out
        fake = _FakeBridge()
        mgr._bridge = fake

        mgr.add_link(a, b, "out", "val")  # must NOT raise

        # edited instance got the live link + the def carries the local-form link
        assert {"node_out": a, "node_in": b, "slot_out": "out", "slot_in": "val"} in mgr._links
        assert {"node_out": la, "node_in": lb, "slot_out": "out", "slot_in": "val"} in mgr._definitions[def_id].links
        assert any(
            e.get("payload", {}).get("node") == inst2 and "mirror" in e["payload"]["error"].lower()
            for e in fake.control.errors
        ), f"link mirror failure not surfaced; got: {fake.control.errors}"
    finally:
        mgr._bridge = None
        mgr.terminate(notify_gui=False)


def test_set_expression_on_a_node_member_of_a_shared_subpatch_with_a_nested_instance():
    """set_expression builds a local->display name map over ALL members to rewrite nd()
    refs. A shared sub-patch may hold a NESTED INSTANCE as a member (lives in _instances,
    not nodes), so the map must resolve names via _entity_name — not self.nodes, which
    KeyErrors on the instance uid and leaves the shared family desynced mid-mirror."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        buf = mgr.add_node("Buffer", "signal")
        inner = mgr.group_nodes([buf])  # a nested instance
        outer = mgr.group_nodes([osc, inner])  # holds a node (osc) + a nested instance
        def_id = mgr.share_instance(outer)
        inst2 = mgr.instantiate_definition(def_id)  # a sibling to mirror into
        local_osc = mgr._instances[outer].members[osc]

        # Must NOT raise (used to KeyError on the nested-instance member uid).
        mgr.set_expression(osc, "oscillator", "frequency", "5 + 5", enabled=True)

        # primary applied + definition recorded + sibling mirrored (strict mirror intact)
        assert mgr.nodes[osc].params["oscillator"]["frequency"].expression == "5 + 5"
        assert mgr._definitions[def_id].members[local_osc]["params"]["oscillator"]["frequency"]["expression"] == "5 + 5"
        sib_osc = mgr._member_uid(inst2, local_osc)
        assert mgr.nodes[sib_osc].params["oscillator"]["frequency"].expression == "5 + 5"
    finally:
        mgr.terminate(notify_gui=False)


def test_deleting_one_shared_instance_keeps_a_sibling_chained_boundary_intact():
    """Deleting one instance of a shared family must leave its siblings + the definition
    untouched. The recursive teardown defensively unwires parent boundaries that forward
    into each removed member; for a SHARED parent (the instance being deleted) that unwire
    must NOT mirror to surviving siblings — else deleting P1 silently unwires P2's chained
    boundary (and drops its external links) and poisons the definition."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        leaf = mgr.add_node("Oscillator", "inputs")
        inner = mgr.group_nodes([leaf])  # nested sub-patch C
        outer = mgr.group_nodes([inner])  # P holds C as a nested-instance member
        bP = mgr.add_boundary(outer, "out", "ARRAY")
        mgr.wire_boundary_to_leaf(outer, bP, leaf, "out")  # chain P.bP -> C -> leaf.out
        def_P = mgr.share_instance(outer)
        sibling = mgr.instantiate_definition(def_P)  # P2, keeps def_P alive

        # Precondition: the chained boundary is wired on the sibling AND in the definition.
        assert mgr._instances[sibling].interface[bP].inner_node is not None
        assert mgr._definitions[def_P].interface[bP].inner_node is not None

        mgr.remove_instance(outer)  # delete the original P1

        # The sibling survives untouched and the definition is not poisoned.
        assert sibling in mgr._instances
        assert mgr._instances[sibling].interface[bP].inner_node is not None, (
            "deleting one shared instance unwired the sibling's chained boundary"
        )
        assert mgr._definitions[def_P].interface[bP].inner_node is not None, (
            "deleting one shared instance poisoned the definition's interface"
        )
    finally:
        mgr.terminate(notify_gui=False)


def test_removing_a_wired_shared_member_prunes_the_def_link():
    """Deleting a WIRED member of a shared sub-patch must also prune the definition's
    internal member->member link template. Otherwise def.links keeps a dangling local and
    every consumer of def.links (instantiate, and load via _splice_doc) KeyErrors on the
    gone member — so a saved .gfi becomes permanently unloadable (silent data loss)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        x = mgr.add_node("Oscillator", "inputs")
        y = mgr.add_node("Buffer", "signal")
        inst1 = mgr.group_nodes([x, y])
        def_id = mgr.share_instance(inst1)
        mgr.instantiate_definition(def_id)  # a sibling, keeps the def alive
        ly = mgr._instances[inst1].members[y]
        mgr.add_link(x, y, "out", "val")  # mirrors {lx->ly} into def.links
        assert any(link["node_in"] == ly for link in mgr._definitions[def_id].links)

        mgr.remove_node(y)  # routes to _remove_shared_member

        # The def link template no longer references the removed member's local...
        assert not any(
            link["node_out"] == ly or link["node_in"] == ly for link in mgr._definitions[def_id].links
        ), "stale def.link references the deleted member"
        # ...so instantiating the def no longer KeyErrors on the gone local.
        new_inst = mgr.instantiate_definition(def_id)
        assert new_inst in mgr._instances
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_shared_member_mirrors_across_siblings_and_def():
    """Bug C: deleting a member of a SHARED sub-patch is the symmetric inverse of the
    shared ADD — it mirror-removes the member from the definition AND every sibling
    instance (strict mirror), instead of the old 'make it unique first' block. This is
    what makes undo of add-into-a-shared sub-patch work (the add's exact inverse)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        # add a member to the shared family — mirrors into inst2 + the definition
        member = mgr.add_member_node(inst1, "Buffer", "signal")
        local = mgr._instances[inst1].members[member]
        assert local in mgr._definitions[def_id].members
        mirror = mgr._member_uid(inst2, local)
        assert mirror is not None and mirror in mgr.nodes

        # delete it — must NOT raise, and must mirror-remove across the whole family
        mgr.remove_node(member)

        assert member not in mgr.nodes  # the edited instance's copy is gone
        assert mirror not in mgr.nodes  # the sibling's mirror is gone too
        assert local not in mgr._definitions[def_id].members  # def template updated
        assert local not in mgr._instances[inst1].members.values()
        assert local not in mgr._instances[inst2].members.values()
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_wired_shared_member_unwires_boundary_across_family():
    """A shared member wired to a boundary: deleting it must also unwire that boundary in
    the definition AND every sibling — not leave the family's interface dangling at a
    removed local. Mirror-remove is the inverse of the shared add + its boundary wiring."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        member = mgr.add_member_node(inst1, "Oscillator", "inputs")
        local = mgr._instances[inst1].members[member]
        out_slot = list(mgr.nodes[member].output_slots)[0]
        dtype = mgr.nodes[member].output_slots[out_slot].name
        bnd = mgr.add_boundary(inst1, "out", dtype)
        mgr.wire_boundary(inst1, bnd, local, out_slot)
        # sanity: the boundary mirrored to inst2 + the def, wired to the member's local
        assert mgr._instances[inst1].interface[bnd].inner_node == local
        assert mgr._instances[inst2].interface[bnd].inner_node == local
        assert mgr._definitions[def_id].interface[bnd].inner_node == local

        mgr.remove_node(member)  # delete the member → its boundary must unwire everywhere

        assert mgr._instances[inst1].interface[bnd].inner_node is None
        assert mgr._instances[inst2].interface[bnd].inner_node is None
        assert mgr._definitions[def_id].interface[bnd].inner_node is None
    finally:
        mgr.terminate(notify_gui=False)


def test_remove_node_rejects_a_nested_instance_member_of_a_shared_subpatch():
    """Mirror-remove handles NODE members; a nested-instance member of a SHARED sub-patch
    is different (recursive subtree teardown across the family + the def's `instances` map
    is Phase 3d). remove_node must reject it with the SAME 'make the parent unique first'
    policy as remove_instance/_reject_if_in_shared_parent — never crash in _teardown_node
    (instance uids aren't in self.nodes) or silently orphan the definition's nested ref."""
    import pytest

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inner = mgr.group_nodes([a])  # a nested instance
        outer = mgr.group_nodes([inner])  # outer holds the nested instance as a member
        mgr.share_instance(outer)  # outer shared; inner is a nested-instance member
        assert inner in mgr._instances[outer].members  # precondition

        with pytest.raises(ValueError):
            mgr.remove_node(inner)

        # nothing corrupted: the nested instance + its node + the def survive intact
        assert inner in mgr._instances
        assert a in mgr.nodes
        assert inner in mgr._instances[outer].members
    finally:
        mgr.terminate(notify_gui=False)


def test_re_share_reattaches_to_existing_def_reuniting_the_family():
    """Undo of make_unique must re-attach the instance to its ORIGINAL definition (when
    siblings kept it alive), reuniting the strict-mirror family — not mint a fresh def (which
    splits the family) nor spawn an extra sibling (what the old duplicate_shared inverse did)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)  # sibling keeps def_id alive
        mgr.make_unique(inst2)
        assert mgr._instances[inst2].def_id is None
        assert def_id in mgr._definitions

        before = {i for i in mgr._instances}
        result = mgr.re_share_instance(inst2, def_id)

        assert result == def_id  # the SAME def, not a fresh one
        assert mgr._instances[inst2].def_id == def_id
        assert mgr._instances[inst2].kind == "shared"
        assert {i for i in mgr._instances} == before  # no extra sibling spawned
        # the family is reunited: editing inst1's member mirrors to inst2 again
        m1 = next(iter(mgr._instances[inst1].members))
        mgr.update_param(m1, "oscillator", "frequency", 7.0)
        m2 = next(iter(mgr._instances[inst2].members))
        assert mgr.nodes[m2].params["oscillator"]["frequency"].value == 7.0
    finally:
        mgr.terminate(notify_gui=False)


def test_re_share_mints_a_fresh_def_when_the_original_was_gced():
    """make_unique on the LAST instance GC's the definition; re-sharing to that now-gone def
    falls back to minting a fresh one (the instance is alone, so identity doesn't matter)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst)
        mgr.make_unique(inst)  # last instance -> def GC'd
        assert def_id not in mgr._definitions

        result = mgr.re_share_instance(inst, def_id)

        # A fresh definition was minted (the gone one couldn't be re-attached to); the freed
        # id may be reused, but the point is a real def now backs the re-shared instance.
        assert result in mgr._definitions
        assert mgr._instances[inst].def_id == result
        assert mgr._instances[inst].kind == "shared"
    finally:
        mgr.terminate(notify_gui=False)


def test_set_boundary_pos_mirrors_to_def_siblings_and_future_instances():
    """Moving an In/Out pill on one shared instance must mirror the pos to the definition,
    every existing sibling, AND any future instance — the def and instances must stay in
    lockstep. Pins the strict-mirror so the def-side immutability refactor can't regress it."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)
        bnd = mgr.add_boundary(inst1, "out", "array")  # mirrors to def + inst2

        changed = mgr.set_boundary_pos(inst1, bnd, [42.0, 7.0])

        assert (inst1, bnd) in changed and (inst2, bnd) in changed
        assert mgr._instances[inst1].interface[bnd].pos == [42.0, 7.0]
        assert mgr._instances[inst2].interface[bnd].pos == [42.0, 7.0]
        assert mgr._definitions[def_id].interface[bnd].pos == [42.0, 7.0]
        # A freshly-instantiated sibling inherits the moved pos from the def.
        inst3 = mgr.instantiate_definition(def_id)
        assert mgr._instances[inst3].interface[bnd].pos == [42.0, 7.0]
    finally:
        mgr.terminate(notify_gui=False)


def test_re_share_mints_a_fresh_def_when_members_no_longer_match_the_def():
    """A unique instance whose members diverged from the def (a member added/removed while
    unique) must NOT silently re-attach to that def — reattaching would leave the extra
    member absent from the def, so strict-mirror edits to it would never reach the def or
    siblings. re_share falls back to a fresh def capturing the instance's ACTUAL members."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)  # sibling keeps def_id alive
        mgr.make_unique(inst2)
        # Edit inst2 while unique: add a member the def doesn't have.
        mgr.add_member_node(inst2, "Buffer", "signal")
        assert len(mgr._instances[inst2].members) == 2
        assert len(mgr._definitions[def_id].members) == 1  # def still has just the oscillator

        result = mgr.re_share_instance(inst2, def_id)

        assert result != def_id, "re_share silently reattached to a member-mismatched def"
        new_def = mgr._instances[inst2].def_id
        assert new_def == result
        # The fresh def captures inst2's actual members, so strict-mirror holds for them.
        assert set(mgr._definitions[new_def].members.keys()) == set(mgr._instances[inst2].members.values())
        assert mgr._instances[inst2].kind == "shared"
        # inst1's original family is untouched.
        assert len(mgr._definitions[def_id].members) == 1
    finally:
        mgr.terminate(notify_gui=False)
