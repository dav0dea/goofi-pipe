"""Flat global node names (Phase 2a of the recursive-sub-patch redesign).

Every node — top-level or sub-patch member, at any depth — has a single globally
unique FLAT display name; there is no `inst::local` qualification. nd() resolves by
the bare name. A member's `local` is a purely-internal template key, decoupled from
its display name, so:

  - group / expand never rename a member or rewrite an nd() ref (display unchanged);
  - share stores the def's intra-patch nd() refs against the template local;
  - instantiate spawns members with fresh flat names and rewrites those refs to them;
  - a rename is a flat rename that follows nd() refs to the new name.
"""
from .test_manager import _bare_manager, _disp, _member


def test_rename_node_cannot_collide_with_an_instance_label():
    """Nodes and sub-patch instances share ONE flat display namespace (both are
    first-class, both render in the same canvas, nd() resolves by bare name). A node
    rename must not be allowed to take a name already held by an instance, or nd()
    resolution becomes ambiguous and the two entities shadow each other."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        inst_name = mgr._instances[inst].name  # 'subpatch0'
        ext = mgr.add_node("Oscillator", "inputs")
        mgr.rename_node(ext, inst_name)
        # the rename was disambiguated away from the instance's label
        assert mgr.nodes[ext].name != inst_name
        names = [mgr.nodes[u].name for u in mgr.nodes] + [
            i.name for i in mgr._instances.values()
        ]
        assert len(names) == len(set(names))  # globally unique across both namespaces
    finally:
        mgr.terminate(notify_gui=False)


def test_group_keeps_flat_member_display_names():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # oscillator0
        b = mgr.add_node("Buffer", "signal")  # buffer0
        name_a, name_b = mgr.nodes[a].name, mgr.nodes[b].name
        inst = mgr.group_nodes([a, b])
        # display names are UNCHANGED by grouping — no inst:: qualification
        assert mgr.nodes[a].name == name_a
        assert mgr.nodes[b].name == name_b
        assert "::" not in mgr.nodes[a].name
        # the member is still organizationally inside the instance
        assert mgr._membership[a] == inst
    finally:
        mgr.terminate(notify_gui=False)


def test_group_preserves_bare_nd_cross_ref():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        a_name = mgr.nodes[a].name
        mgr.set_expression(b, "oscillator", "frequency", f"nd('{a_name}')", enabled=True)
        mgr.group_nodes([a, b])
        # bare nd ref is untouched — the referenced member's flat name didn't change
        assert mgr.nodes[b].params["oscillator"]["frequency"].expression == f"nd('{a_name}')"
    finally:
        mgr.terminate(notify_gui=False)


def test_expand_does_not_rename_members():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        names = {a: mgr.nodes[a].name, b: mgr.nodes[b].name}
        inst = mgr.group_nodes([a, b])
        mgr.expand_instance(inst)
        assert mgr.nodes[a].name == names[a] and mgr.nodes[b].name == names[b]
    finally:
        mgr.terminate(notify_gui=False)


def test_add_member_gets_a_fresh_flat_name():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        uid = mgr.add_member_node(inst, "Buffer", "signal")
        assert "::" not in mgr.nodes[uid].name
        # globally unique flat name, like any freshly-added node
        names = [mgr.nodes[u].name for u in mgr.nodes]
        assert len(set(names)) == len(names)
    finally:
        mgr.terminate(notify_gui=False)


def test_instantiate_gives_fresh_flat_names_and_rewrites_nd_to_own_members():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # oscillator0
        b = mgr.add_node("Oscillator", "inputs")  # oscillator1
        a_name = mgr.nodes[a].name
        # b references a by flat name
        mgr.set_expression(b, "oscillator", "frequency", f"nd('{a_name}')", enabled=True)
        inst1 = mgr.group_nodes([a, b])
        la, lb = mgr._instances[inst1].members[a], mgr._instances[inst1].members[b]
        def_id = mgr.share_instance(inst1)
        inst2 = mgr.instantiate_definition(def_id)

        a2, b2 = _member(mgr, inst2, la), _member(mgr, inst2, lb)
        # the new instance's members carry fresh flat names (no qualification)
        assert "::" not in mgr.nodes[a2].name and "::" not in mgr.nodes[b2].name
        assert mgr.nodes[a2].name != a_name  # globally unique, not a clone of inst1's
        # b2 references ITS OWN sibling a2 by a2's flat name
        assert mgr.nodes[b2].params["oscillator"]["frequency"].expression == f"nd('{mgr.nodes[a2].name}')"
    finally:
        mgr.terminate(notify_gui=False)


def test_member_nd_cross_ref_is_flat_per_instance_through_save_load(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        mgr.set_expression(b, "oscillator", "frequency", f"nd('{mgr.nodes[a].name}')", enabled=True)
        inst1 = mgr.group_nodes([a, b])
        def_id = mgr.share_instance(inst1)
        mgr.instantiate_definition(def_id)
        fp = str(tmp_path / "flat.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate(notify_gui=False)

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        # in EVERY restored instance, the dependent member references its own sibling
        # by that sibling's flat name (no qualification, no cross-instance leakage)
        for iid, inst in mgr2._instances.items():
            local_to_uid = {l: u for u, l in inst.members.items()}
            # find the member whose freq is expression-bound
            for uid, local in inst.members.items():
                expr = mgr2.nodes[uid].params["oscillator"]["frequency"].expression
                if expr:
                    assert "::" not in expr
                    # it points at a real, live node in THIS instance
                    target = expr[len("nd('"):-len("')")]
                    assert any(mgr2.nodes[u].name == target for u in inst.members)
    finally:
        mgr2.terminate(notify_gui=False)


def test_member_uid_raises_on_duplicate_local_rather_than_silent_misroute():
    """A member's local is a per-instance template key; if two members ever share one
    (corruption), _member_uid must fail loudly instead of returning whichever it scans
    first and silently splicing boundaries/data onto the wrong member."""
    import pytest

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([a, b])
        for uid in list(mgr._instances[inst].members):
            mgr._instances[inst].members[uid] = "dup"  # force a collision
        with pytest.raises(RuntimeError):
            mgr._member_uid(inst, "dup")
    finally:
        mgr.terminate(notify_gui=False)


def test_rename_member_rewrites_nd_refs_to_new_flat_name():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # member
        ext = mgr.add_node("Oscillator", "inputs")  # external referrer
        mgr.set_expression(ext, "oscillator", "frequency", f"nd('{mgr.nodes[a].name}')", enabled=True)
        inst = mgr.group_nodes([a])
        mgr.rename_node(a, "carrier")
        assert mgr.nodes[a].name == "carrier"
        # the external referrer's nd() ref followed to the new flat name
        assert mgr.nodes[ext].params["oscillator"]["frequency"].expression == "nd('carrier')"
    finally:
        mgr.terminate(notify_gui=False)
