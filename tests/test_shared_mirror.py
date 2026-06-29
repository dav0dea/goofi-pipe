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
