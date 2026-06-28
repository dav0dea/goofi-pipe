"""Node identity contract: `uid` is the universal key, `name` is mutable display-only.

The manager keys its whole graph (NodeContainer, links, membership, groups) on a
stable per-node `uid`. The display `name` lives ON the node and can be renamed
freely without touching any reference — renaming is safe BY CONSTRUCTION because
nothing keys on the name.
"""
from .test_manager import _bare_manager


def test_add_node_returns_uid_distinct_from_display_name():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        uid = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[uid]  # container is keyed by uid
        assert ref.uid == uid
        assert ref.name == "oscillator0"  # display name still auto-numbered
        assert uid != ref.name  # the uid is NOT the display name
    finally:
        mgr.terminate(notify_gui=False)


def test_links_reference_uid_not_name():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.add_link(a, b, "out", "val")
        assert {"node_out": a, "node_in": b, "slot_out": "out", "slot_in": "val"} in mgr._links
    finally:
        mgr.terminate(notify_gui=False)


def test_rename_changes_display_name_only_uid_and_links_stable():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.add_link(a, b, "out", "val")

        mgr.rename_node(a, "my_oscillator")

        assert mgr.nodes[a].name == "my_oscillator"  # display changed
        assert a in mgr.nodes  # uid is stable across rename
        # the link still references the SAME uid — no cascade needed
        assert {"node_out": a, "node_in": b, "slot_out": "out", "slot_in": "val"} in mgr._links
    finally:
        mgr.terminate(notify_gui=False)


def test_default_names_stay_auto_numbered_per_type():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        assert mgr.nodes[a].name == "buffer0"
        assert mgr.nodes[b].name == "buffer1"
        assert a != b  # distinct uids
    finally:
        mgr.terminate(notify_gui=False)


def test_rename_to_a_taken_name_is_disambiguated_so_nd_stays_unique():
    # nd('name') resolves BY display name, so two live nodes must never share one.
    # rename auto-disambiguates a collision (consistent with add's auto-numbering).
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")  # oscillator0
        b = mgr.add_node("Buffer", "signal")  # buffer0
        mgr.rename_node(b, "oscillator0")  # collides with a
        assert mgr.nodes[a].name == "oscillator0"
        assert mgr.nodes[b].name != "oscillator0"  # disambiguated
        # the nd() directory has a distinct entry for every live node
        directory = mgr._node_directory()
        assert len(directory) == 2
        assert directory[mgr.nodes[a].name] == mgr.nodes[a].node_id
        assert directory[mgr.nodes[b].name] == mgr.nodes[b].node_id
    finally:
        mgr.terminate(notify_gui=False)


def test_group_dedupes_duplicate_member_local_names():
    # A member's LOCAL name is a key (template + members map + save). Even if two
    # members somehow carry the same display name, grouping must keep locals unique
    # so neither member is silently dropped on save.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        mgr.nodes[b].name = mgr.nodes[a].name  # force a duplicate display name
        inst = mgr.group_nodes([a, b])
        locals_ = list(mgr._instances[inst]["members"].values())
        assert len(set(locals_)) == 2, f"member locals collided: {locals_}"
        # both members persist (the duplicate-local collapse would drop one) — the
        # v2 tree keys unique-instance members by local name.
        _root_nodes, _root_links, _defs, instances = mgr.build_v2_tree()
        assert len(instances[inst]["members"]) == 2
    finally:
        mgr.terminate(notify_gui=False)


def test_rename_subpatch_member_renames_its_local_and_survives_roundtrip(tmp_path):
    # Renaming a member via the qualified display name re-keys its LOCAL name (the
    # one source of truth the canvas + save/load read), not just the label.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        u = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([u])
        from goofi.manager import SUBPATCH_SEP

        old_local = mgr._instances[inst]["members"][u]
        mgr.rename_node(u, f"{inst}{SUBPATCH_SEP}custom")
        assert mgr._instances[inst]["members"][u] == "custom"
        assert mgr.nodes[u].membership["local_name"] == "custom"
        assert mgr.nodes[u].name == f"{inst}{SUBPATCH_SEP}custom"
        assert old_local != "custom"
        fp = str(tmp_path / "m.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate(notify_gui=False)

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        inst2 = next(iter(mgr2._instances))
        assert "custom" in mgr2._instances[inst2]["members"].values()
    finally:
        mgr2.terminate(notify_gui=False)


def test_rename_shared_member_is_rejected():
    # Renaming a member of a SHARED instance would have to mirror across siblings +
    # the definition (strict mirror); until that exists, reject it cleanly rather
    # than leave inconsistent state.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        u = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([u])
        mgr.share_instance(inst)  # promotes inst to shared (sets def_id)
        member = next(iter(mgr._instances[inst]["members"]))
        from goofi.manager import SUBPATCH_SEP

        import pytest

        with pytest.raises(Exception):
            mgr.rename_node(member, f"{inst}{SUBPATCH_SEP}renamed")
    finally:
        mgr.terminate(notify_gui=False)


def test_rename_then_save_load_keeps_uid_and_new_display_name(tmp_path):
    # The headline scenario: a renamed node round-trips with BOTH its stable uid
    # and the new display name (the .gfi persists the name inside the uid record).
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        mgr.add_link(a, b, "out", "val")
        mgr.rename_node(a, "brain_wave")
        fp = str(tmp_path / "r.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate(notify_gui=False)

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        assert a in mgr2.nodes  # uid stable across save/load
        assert mgr2.nodes[a].name == "brain_wave"  # renamed display survived
        # the link still references the same uids
        assert {"node_out": a, "node_in": b, "slot_out": "out", "slot_in": "val"} in mgr2._links
    finally:
        mgr2.terminate(notify_gui=False)


def test_restart_preserves_uid_and_display_name():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        mgr.rename_node(a, "keep_me")
        old_node_id = mgr.nodes[a].node_id
        mgr.restart_node(a)
        assert a in mgr.nodes  # uid is stable across restart
        assert mgr.nodes[a].name == "keep_me"  # display preserved
        assert mgr.nodes[a].node_id != old_node_id  # transport id is fresh
    finally:
        mgr.terminate(notify_gui=False)


def test_bridge_snapshot_carries_uid_and_display_name():
    # The wire contract (Phase 2): the node description sent to the browser keys on
    # `uid` (identity) and carries the mutable `name` (display) separately.
    from goofi.bridge.schemas import describe_node_instance

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        uid = mgr.add_node("Oscillator", "inputs")
        desc = describe_node_instance(uid, mgr.nodes[uid])
        assert desc["uid"] == uid
        assert desc["name"] == "oscillator0"
        # rename moves only the display field; the uid is stable
        mgr.rename_node(uid, "my_osc")
        desc2 = describe_node_instance(uid, mgr.nodes[uid])
        assert desc2["uid"] == uid and desc2["name"] == "my_osc"
    finally:
        mgr.terminate(notify_gui=False)
