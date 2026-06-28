"""Tests for the Manager after the iceoryx2 refactor."""
import time
import uuid
from os import path

import pytest

import goofi
from dataclasses import asdict

from goofi.manager import Boundary, Manager, NodeContainer
from goofi.transport import (
    WaitSet,
    open_subscriber,
    set_instance_id,
)

MANAGER_TEST_DURATION = 0.2


def _bare_manager(use_multiprocessing: bool = True) -> Manager:
    """Construct a Manager without entering its blocking event loop."""
    import os, atexit
    from goofi.manager import _cleanup_iceoryx2_shm
    from goofi.node_helpers import NodeProcessRegistry, list_nodes

    mgr = Manager.__new__(Manager)
    instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    set_instance_id(instance_id)
    atexit.register(_cleanup_iceoryx2_shm, instance_id)
    list_nodes(verbose=False)
    mgr._instance_id = instance_id
    mgr._headless = True
    mgr._use_multiprocessing = use_multiprocessing
    mgr._running = True
    mgr.nodes = NodeContainer()
    mgr._node_groups = {}
    mgr._links = []
    mgr._refs_by_uid = {}
    mgr._membership = {}
    mgr._instances = {}
    mgr._definitions = {}
    NodeProcessRegistry().headless = True
    mgr._save_path = None
    mgr._unsaved_changes = False
    mgr._bridge = None
    return mgr


def _build_simple_graph(mgr: Manager) -> tuple[str, str]:
    """Oscillator → Select with a 0:5 include filter. Returns their UIDs."""
    osc = mgr.add_node("Oscillator", "inputs")
    sel = mgr.add_node("Select", "array", params={"select": {"include": "0:5"}})
    mgr.add_link(osc, sel, "out", "data")
    return osc, sel


def _member(mgr: Manager, inst: str, local: str) -> str:
    """The live UID of the member with local name `local` in instance `inst`."""
    for uid, l in mgr._instances[inst].members.items():
        if l == local:
            return uid
    raise KeyError(f"{inst}:{local}")


def _disp(mgr: Manager, uid: str) -> str:
    """The display name of a node by uid."""
    return mgr.nodes[uid].name


def test_serialize_patch_returns_yaml_without_writing(tmp_path):
    import yaml as _yaml
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _build_simple_graph(mgr)
        text = mgr.serialize_patch()
        doc = _yaml.load(text, Loader=_yaml.FullLoader)
        # v2 envelope: nodes/links live under root; version + (empty) sub-patch maps.
        assert doc["version"] == 2
        assert doc["definitions"] == {} and doc["root"]["instances"] == {}
        assert len(doc["root"]["nodes"]) == 2
        # every node carries a stable uid
        assert all("uid" in n for n in doc["root"]["nodes"].values())
        # serialize must not have created a file or set save_path
        assert mgr.save_path is None
    finally:
        mgr.terminate()


def test_save_load_roundtrip_preserves_uids(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, sel = _build_simple_graph(mgr)
        uids_before = {osc, sel}  # add_node returns the uid (the container key)
        assert all(mgr.nodes[u].uid == u for u in (osc, sel))
        fp = str(tmp_path / "p.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        assert set(mgr2.nodes) == uids_before  # uid is stable across save/load
        assert all(mgr2.nodes[u].uid == u for u in uids_before)
    finally:
        mgr2.terminate()


def test_node_record_reflects_live_params_despite_stale_serialized_state():
    # update_param / set_expression write ref.params synchronously, but
    # serialized_state only refreshes on the node's next echo. A save or a sub-patch
    # share immediately after an edit must still capture the live binding — _node_record
    # reads the authoritative live params, not the (possibly stale) snapshot.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        mgr.set_expression(b, "oscillator", "frequency", f"nd('{mgr.nodes[a].name}')", enabled=True)
        # serialized_state still holds the pre-edit value...
        assert mgr.nodes[b].serialized_state["params"]["oscillator"]["frequency"] != \
            mgr.nodes[b].params["oscillator"]["frequency"].expression
        # ...but the record captures the live expression.
        rec = mgr._node_record(b)
        assert rec["params"]["oscillator"]["frequency"]["expression"] == f"nd('{mgr.nodes[a].name}')"
    finally:
        mgr.terminate(notify_gui=False)


def _build_grouped_graph(mgr):
    """osc → sel0 → sel1, with [sel0, sel1] grouped into one sub-patch.

    Returns (osc_uid, inst_id). Member uids are reached via `_member(mgr, inst, local)`.
    """
    osc = mgr.add_node("Oscillator", "inputs")
    sel0 = mgr.add_node("Select", "array", params={"select": {"include": "0:5"}})
    sel1 = mgr.add_node("Select", "array", params={"select": {"include": "0:2"}})
    out0 = list(mgr.nodes[sel0].output_slots)[0]
    mgr.add_link(osc, sel0, "out", "data")
    mgr.add_link(sel0, sel1, out0, "data")
    inst_id = mgr.group_nodes([sel0, sel1])
    return osc, inst_id


def test_group_nodes_namespaces_members_and_records_state():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        assert mgr._instances[inst].name == "subpatch0"  # display label (the key is a uid)
        s0, s1 = _member(mgr, inst, "select0"), _member(mgr, inst, "select1")
        # flat naming: members keep their (unchanged, globally-unique) display names
        assert _disp(mgr, s0) == "select0"
        assert _disp(mgr, s1) == "select1"
        assert mgr._membership[s0] == inst
        # external link is unchanged (it references the member uid, not the name)
        assert {"node_out": osc, "node_in": s0, "slot_out": "out", "slot_in": "data"} in [
            dict(link) for link in mgr.links
        ]
        # interface derived: select0.data is a boundary input (fed by osc)
        assert "select0.data" in mgr._instances[inst].interface
    finally:
        mgr.terminate()


def test_subpatch_save_load_roundtrip_and_expand(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        fp = str(tmp_path / "grouped.gfi")
        mgr.save(fp, overwrite=True)
        # the v2 file carries the instance, not flat members
        import yaml as _yaml
        doc = _yaml.load(open(fp), Loader=_yaml.FullLoader)
        assert inst in doc["root"]["instances"]  # keyed by the instance's stable uid
        # the saved instance is template-form: members keyed by LOCAL name
        assert set(doc["root"]["instances"][inst]["members"]) == {"select0", "select1"}
        assert len(doc["root"]["instances"][inst]["links"]) == 1  # internal sel0->sel1
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        assert inst in mgr2._instances  # the instance uid persists across save/load
        s0, s1 = _member(mgr2, inst, "select0"), _member(mgr2, inst, "select1")
        assert s0 in mgr2.nodes and s1 in mgr2.nodes
        assert mgr2._membership[s1] == inst
        assert _disp(mgr2, s0) == "select0"  # flat name round-trips
        # both the external and internal links are restored (by uid)
        links = [dict(link) for link in mgr2.links]
        assert {"node_out": osc, "node_in": s0, "slot_out": "out", "slot_in": "data"} in links
        assert any(l["node_out"] == s0 and l["node_in"] == s1 for l in links)

        # expand dissolves the group back to bare display names (uids unchanged)
        restored = mgr2.expand_instance(inst)
        assert set(restored) == {s0, s1}  # returns member uids
        assert {_disp(mgr2, u) for u in restored} == {"select0", "select1"}
        assert mgr2._instances == {} and mgr2._membership == {}
        links2 = [dict(link) for link in mgr2.links]
        assert {"node_out": osc, "node_in": s0, "slot_out": "out", "slot_in": "data"} in links2
    finally:
        mgr2.terminate()


def test_shared_param_edit_mirrors_to_sibling():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # unique sub-patch with select0, select1
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        # both instances reference the same definition
        assert mgr._instances[inst].def_id == def_id
        assert mgr._instances[inst2].def_id == def_id
        # edit a param on instance A's select0 → mirrors to instance B's select0
        mgr.update_param(_member(mgr, inst, "select0"), "select", "include", "3:9")
        assert mgr.nodes[_member(mgr, inst2, "select0")].params["select"]["include"].value == "3:9"
        # the definition (save source of truth) also reflects the edit
        assert mgr._definitions[def_id].members["select0"]["params"]["select"]["include"] == "3:9"
    finally:
        mgr.terminate()


def test_add_member_node_lands_in_subpatch():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # unique sub-patch select0, select1
        uid = mgr.add_member_node(inst, "Buffer", "signal")
        # flat naming: a fresh globally-unique display name, no inst:: qualification
        assert _disp(mgr, uid) == "buffer0"
        assert uid in mgr.nodes
        assert mgr._membership[uid] == inst
        assert mgr._instances[inst].members[uid] == "buffer0"  # local template key
        # the node carries the membership marker the bridge/save read
        assert mgr.nodes[uid].membership == {"instance": inst, "local_name": "buffer0"}
        # a fresh add picks the next free flat name
        uid2 = mgr.add_member_node(inst, "Buffer", "signal")
        assert _disp(mgr, uid2) == "buffer1"
    finally:
        mgr.terminate()


def test_add_member_node_mirrors_to_shared_siblings():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        # add a member to instance A → mirrors into the def and sibling B
        uid = mgr.add_member_node(inst, "Buffer", "signal")
        local = mgr._instances[inst].members[uid]
        assert local in mgr._definitions[def_id].members
        sib_uid = _member(mgr, inst2, local)
        assert sib_uid in mgr.nodes
        assert mgr._instances[inst2].members[sib_uid] == local
        # both families stay strict mirrors: same member locals on each side
        assert set(mgr._instances[inst].members.values()) == set(
            mgr._instances[inst2].members.values()
        )
    finally:
        mgr.terminate()


def test_output_boundary_resolves_for_data_route():
    # The collapsed sub-patch's output viewer opens /data/<inst>/<boundary>; the
    # data route splices that via resolve_boundary to the inner member's real
    # (uid, slot). Verify that mapping lands on a streaming node + output slot.
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # subpatch0 with select0, select1
        s1 = _member(mgr, inst, "select1")
        out_slot = list(mgr.nodes[s1].output_slots)[0]
        bnd = mgr.add_boundary(inst, "out", "ARRAY")
        mgr.wire_boundary(inst, bnd, "select1", out_slot)
        node, slot = mgr.resolve_boundary(inst, bnd)
        assert node == s1 and slot == out_slot
        # data.py's preconditions: the resolved node is real and has that output.
        assert node in mgr.nodes
        assert slot in mgr.nodes[node].output_slots
    finally:
        mgr.terminate()


def test_make_unique_detaches_and_gcs_definition():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.make_unique(inst2)
        assert mgr._instances[inst2].def_id is None
        assert mgr._instances[inst2].kind == "unique"
        # def still referenced by `inst`
        assert def_id in mgr._definitions
        # editing inst2 no longer affects inst
        mgr.update_param(_member(mgr, inst2, "select0"), "select", "include", "7:8")
        assert mgr.nodes[_member(mgr, inst, "select0")].params["select"]["include"].value != "7:8"
        # make the original unique too → definition is GC'd
        mgr.make_unique(inst)
        assert def_id not in mgr._definitions
    finally:
        mgr.terminate()


def _build_single_member_subpatch(mgr):
    """Group a lone Buffer into a sub-patch with an (initially) empty interface."""
    buf = mgr.add_node("Buffer", "signal")
    inst = mgr.group_nodes([buf])
    return buf, inst


def test_add_and_wire_boundary():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)  # buffer0 -> inst::buffer0
        assert mgr._instances[inst].interface == {}
        bid = mgr.add_boundary(inst, "in", "ARRAY", pos=(5, 6))
        e = mgr._instances[inst].interface[bid]
        assert asdict(e) == {"dir": "in", "dtype": "ARRAY", "inner_node": None, "inner_slot": None, "pos": [5, 6]}
        mgr.wire_boundary(inst, bid, "buffer0", "val")
        e = mgr._instances[inst].interface[bid]
        assert e.inner_node == "buffer0" and e.inner_slot == "val"
        assert mgr.resolve_boundary(inst, bid) == (_member(mgr, inst, "buffer0"), "val")
    finally:
        mgr.terminate()


def test_wire_boundary_rejects_dtype_mismatch():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        bid = mgr.add_boundary(inst, "in", "STRING")
        with pytest.raises(ValueError):
            mgr.wire_boundary(inst, bid, "buffer0", "val")  # val is ARRAY
    finally:
        mgr.terminate()


def test_wire_boundary_dedup_one_per_inner_slot():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        b0 = mgr.add_boundary(inst, "in", "ARRAY")
        mgr.wire_boundary(inst, b0, "buffer0", "val")
        b1 = mgr.add_boundary(inst, "in", "ARRAY")
        with pytest.raises(ValueError):
            mgr.wire_boundary(inst, b1, "buffer0", "val")  # already exposed by b0
    finally:
        mgr.terminate()


def test_wire_in_boundary_rejects_internally_fed_input():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        mgr.add_link(b0, b1, "out", "val")  # internal feed buffer0.out -> buffer1.val
        inst = mgr.group_nodes([b0, b1])
        bid = mgr.add_boundary(inst, "in", "ARRAY")
        with pytest.raises(ValueError):
            mgr.wire_boundary(inst, bid, "buffer1", "val")  # already fed internally
    finally:
        mgr.terminate()


def test_delete_unique_member_cleans_up_and_unwires():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([b0, b1])
        bid = mgr.add_boundary(inst, "in", "ARRAY")
        mgr.wire_boundary(inst, bid, "buffer1", "val")
        b1 = _member(mgr, inst, "buffer1")
        mgr.remove_node(b1)
        # Member dropped from the instance + membership; boundary unwired (not dangling).
        assert b1 not in mgr._instances[inst].members
        assert b1 not in mgr._membership
        assert mgr._instances[inst].interface[bid].inner_node is None
        # Save no longer references the gone member.
        mgr.serialize_patch()
    finally:
        mgr.terminate()


def test_delete_shared_member_blocked():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        def_id = mgr.share_instance(inst)
        mgr.instantiate_definition(def_id)
        with pytest.raises(ValueError):
            mgr.remove_node(_member(mgr, inst, "buffer0"))  # shared member — make unique first
    finally:
        mgr.terminate()


def test_wire_legacy_entry_without_dtype_heals():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        # A legacy interface entry (pre-dtype shape, dtype=None) must not crash on wire.
        mgr._instances[inst].interface["legacy0"] = Boundary(
            dir="in", dtype=None, inner_node=None, inner_slot=None, pos=[0, 0]
        )
        mgr.wire_boundary(inst, "legacy0", "buffer0", "val")
        assert mgr._instances[inst].interface["legacy0"].dtype == "ARRAY"
    finally:
        mgr.terminate()


def test_external_splice_and_unsplice():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        buf = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([buf])
        bid = mgr.add_boundary(inst, "in", "ARRAY")
        mgr.wire_boundary(inst, bid, "buffer0", "val")
        uid, slot = mgr.resolve_boundary(inst, bid)  # what the bridge resolves to (uid)
        mgr.add_link(osc, uid, "out", slot)          # the spliced flat link
        assert any(l["node_out"] == osc and l["node_in"] == uid and l["slot_in"] == slot for l in mgr.links)
        # Deleting the boundary tears down its external wire.
        mgr.remove_boundary(inst, bid)
        assert bid not in mgr._instances[inst].interface
        assert not any(l["node_in"] == uid and l["slot_in"] == slot for l in mgr.links)
    finally:
        mgr.terminate()


def test_rewire_boundary_resplices_external_link():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        b0 = mgr.add_node("Buffer", "signal")
        b1 = mgr.add_node("Buffer", "signal")
        inst = mgr.group_nodes([b0, b1])  # inst::buffer0, inst::buffer1
        bid = mgr.add_boundary(inst, "in", "ARRAY")
        mgr.wire_boundary(inst, bid, "buffer0", "val")
        b0, b1 = _member(mgr, inst, "buffer0"), _member(mgr, inst, "buffer1")
        mgr.add_link(osc, b0, "out", "val")  # external splice to buffer0
        # Re-wire the In node to buffer1 → its external link follows.
        mgr.wire_boundary(inst, bid, "buffer1", "val")
        assert any(l["node_out"] == osc and l["node_in"] == b1 and l["slot_in"] == "val" for l in mgr.links)
        assert not any(l["node_in"] == b0 and l["node_out"] == osc for l in mgr.links)
    finally:
        mgr.terminate()


def test_shared_boundary_mirrors_and_make_unique_isolates():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        bid = mgr.add_boundary(inst, "in", "ARRAY")  # mirrors to def + sibling
        assert bid in mgr._instances[inst2].interface
        assert bid in mgr._definitions[def_id].interface
        mgr.wire_boundary(inst, bid, "buffer0", "val")  # inner mirrors to sibling's own member
        assert mgr._instances[inst2].interface[bid].inner_node == "buffer0"
        # Detach inst2; editing it must not cross-mutate inst (deep-copied interface).
        mgr.make_unique(inst2)
        mgr.wire_boundary(inst2, bid, None, None)
        assert mgr._instances[inst2].interface[bid].inner_node is None
        assert mgr._instances[inst].interface[bid].inner_node == "buffer0"
    finally:
        mgr.terminate()


def test_unwired_boundary_survives_save_load(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        bid = mgr.add_boundary(inst, "in", "ARRAY", pos=(7, 8))
        fp = str(tmp_path / "unwired.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        e = mgr2._instances[inst].interface[bid]
        assert e.inner_node is None and e.dtype == "ARRAY" and list(e.pos) == [7, 8]
    finally:
        mgr2.terminate()


def test_shared_member_pos_mirrors_to_sibling():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        a0, b0 = _member(mgr, inst, "select0"), _member(mgr, inst2, "select0")
        # Move a member inside instance A → mirrors to B's corresponding member.
        changed = mgr.set_node_pos(a0, (123, 456))
        assert list(mgr.nodes[a0].gui_kwargs["pos"]) == [123, 456]
        assert list(mgr.nodes[b0].gui_kwargs["pos"]) == [123, 456]
        # Definition (save source of truth) records the pos too.
        assert list(mgr._definitions[def_id].members["select0"]["gui_kwargs"]["pos"]) == [123, 456]
        # Both the moved node and its sibling are reported changed (for broadcasts).
        assert set(changed) == {a0, b0}
        # A non-mirrored member (select1) is untouched in the sibling.
        assert mgr.nodes[_member(mgr, inst2, "select1")].gui_kwargs.get("pos") != [123, 456]
    finally:
        mgr.terminate()


def test_remove_instance_deletes_members_and_gcs_def():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        a0 = _member(mgr, inst, "select0")
        # Delete one shared instance → its members + links gone; def survives (inst2 holds it).
        mgr.remove_instance(inst)
        assert inst not in mgr._instances
        assert a0 not in mgr.nodes
        assert a0 not in mgr._membership
        assert all(link["node_in"] != a0 for link in mgr.links)
        assert def_id in mgr._definitions
        # Delete the last instance → orphaned definition is GC'd.
        mgr.remove_instance(inst2)
        assert def_id not in mgr._definitions
        assert mgr._instances == {}
        assert all(mgr._membership.get(u) != inst2 for u in mgr._membership)
    finally:
        mgr.terminate()


def test_shared_member_pos_survives_save_load(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.set_node_pos(_member(mgr, inst, "select0"), (321, 654))
        fp = str(tmp_path / "shared_pos.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        # The mirrored layout round-trips: every shared sibling's member is at the moved pos.
        shared_ids = [iid for iid, i in mgr2._instances.items() if i.def_id == def_id]
        assert len(shared_ids) == 2
        for iid in shared_ids:
            assert list(mgr2.nodes[_member(mgr2, iid, "select0")].gui_kwargs["pos"]) == [321, 654]
    finally:
        mgr2.terminate()


def test_unique_member_pos_does_not_mirror():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # unique sub-patch
        s0 = _member(mgr, inst, "select0")
        changed = mgr.set_node_pos(s0, (10, 20))
        assert list(mgr.nodes[s0].gui_kwargs["pos"]) == [10, 20]
        assert changed == [s0]
    finally:
        mgr.terminate()


def test_make_unique_stops_pos_mirror():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.make_unique(inst2)
        s0 = _member(mgr, inst, "select0")
        changed = mgr.set_node_pos(s0, (7, 8))
        # inst2 detached → its member must not move with inst's.
        assert mgr.nodes[_member(mgr, inst2, "select0")].gui_kwargs.get("pos") != [7, 8]
        assert changed == [s0]
    finally:
        mgr.terminate()


def test_shared_subpatch_save_load_roundtrip(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        fp = str(tmp_path / "shared.gfi")
        mgr.save(fp, overwrite=True)
        import yaml as _yaml
        doc = _yaml.load(open(fp), Loader=_yaml.FullLoader)
        # the definition is emitted once; both instances reference it as shared
        assert def_id in doc["definitions"]
        shared = [i for i in doc["root"]["instances"].values() if i["kind"] == "shared"]
        assert len(shared) == 2
        assert all(i["def"] == def_id for i in shared)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        # both shared instances restored, members spawned, still mirror on edit
        shared_ids = [iid for iid, i in mgr2._instances.items() if i.def_id == def_id]
        assert len(shared_ids) == 2
        a, b = shared_ids
        mgr2.update_param(_member(mgr2, a, "select0"), "select", "include", "1:4")
        assert mgr2.nodes[_member(mgr2, b, "select0")].params["select"]["include"].value == "1:4"
    finally:
        mgr2.terminate()


def test_creation_smoke():
    """Manager(duration=...) ought to start, run briefly, and shut down cleanly."""
    Manager(duration=MANAGER_TEST_DURATION)


def test_main_entrypoint():
    """The CLI entry point should accept --headless and exit cleanly."""
    goofi.manager.main(MANAGER_TEST_DURATION, ["--headless"])


def test_simple_chain_dataflow():
    """End-to-end: data passes through a 2-node chain via iceoryx2."""
    mgr = _bare_manager()
    try:
        osc, sel = _build_simple_graph(mgr)

        # External subscriber acting like a GUI viewer. The data service is
        # keyed on the node's stable transport id, not its display name.
        service = mgr.nodes[sel].data_service_for("out")
        sub, listener = open_subscriber(service, in_process=False, latest_wins=True)
        ws = WaitSet()
        ws.attach(listener)
        mgr.nodes[sel].register_subscriber("out")

        from goofi.codec import decode_data

        received = []
        deadline = time.time() + 3.0
        while time.time() < deadline and len(received) < 5:
            if ws.wait(0.25):
                buf = sub.take_latest()
                if buf is not None:
                    received.append(decode_data(buf))
        assert len(received) >= 1, "no data reached the downstream subscriber"
        # The Select node was configured with include=0:5, so the latest
        # frame must have at most 5 elements along axis 0.
        assert received[-1].data.shape[0] == 5
    finally:
        mgr.terminate(notify_gui=False)


def test_save_empty(tmpdir):
    mgr = _bare_manager()
    try:
        mgr.save(path.join(str(tmpdir), "test.gfi"))
        assert path.exists(path.join(str(tmpdir), "test.gfi"))
    finally:
        mgr.terminate(notify_gui=False)


def test_save_extension(tmpdir):
    mgr = _bare_manager()
    try:
        _build_simple_graph(mgr)
        mgr.save(path.join(str(tmpdir), "no_extension"))
        assert path.exists(path.join(str(tmpdir), "no_extension.gfi"))
    finally:
        mgr.terminate(notify_gui=False)


@pytest.mark.parametrize("overwrite", [True, False])
def test_save_overwrite(overwrite, tmpdir):
    mgr = _bare_manager()
    tmpdir = str(tmpdir)
    try:
        _build_simple_graph(mgr)
        target = path.join(tmpdir, "test.gfi")
        mgr.save(target, overwrite=overwrite)
        assert path.exists(target)
        if overwrite:
            mgr.save(target, overwrite=True)
            assert path.exists(target)
        else:
            with pytest.raises(FileExistsError):
                mgr.save(target)
    finally:
        mgr.terminate(notify_gui=False)


def test_save_to_directory(tmpdir):
    mgr = _bare_manager()
    tmpdir = str(tmpdir)
    try:
        _build_simple_graph(mgr)
        with pytest.raises(ValueError):
            mgr.save(123)  # not a str
        mgr.save(tmpdir)
        assert path.exists(path.join(tmpdir, "untitled0.gfi"))
        mgr.save(tmpdir)
        assert path.exists(path.join(tmpdir, "untitled1.gfi"))
    finally:
        mgr.terminate(notify_gui=False)


def test_reused_display_name_gets_unique_node_id():
    """Deleting a node frees its display name for reuse, but the replacement
    must get a fresh transport id so its iceoryx2 services never collide with
    the still-shutting-down old node's (which would raise
    ExceedsMaxSupportedPublishers)."""
    mgr = _bare_manager()
    try:
        n1 = mgr.add_node("Oscillator", "inputs")
        id1 = mgr.nodes[n1].node_id
        name1 = mgr.nodes[n1].name
        mgr.remove_node(n1)
        n2 = mgr.add_node("Oscillator", "inputs")
        id2 = mgr.nodes[n2].node_id
        # The display name is reused (nice UX); uid + transport id are not.
        assert name1 == mgr.nodes[n2].name == "oscillator0"
        assert n1 != n2  # fresh uid
        assert id1 != id2
    finally:
        mgr.terminate(notify_gui=False)


def test_node_directory_maps_display_names_to_unique_ids():
    """The manager's name->node_id directory (pushed to nodes so `nd('name')`
    expressions resolve to the producer's stable id) maps each live display
    name to its node's unique transport id."""
    mgr = _bare_manager()
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Oscillator", "inputs")
        directory = mgr._node_directory()  # keyed by DISPLAY name
        assert directory[mgr.nodes[a].name] == mgr.nodes[a].node_id
        assert directory[mgr.nodes[b].name] == mgr.nodes[b].node_id
        assert mgr.nodes[a].node_id != mgr.nodes[b].node_id
    finally:
        mgr.terminate(notify_gui=False)


def test_group_mode_name_reuse_no_clash():
    """Rapid remove + re-add of a node hosted in a shared process group must
    not raise: the host process outlives the node, so the previous instance's
    endpoints must be released AND the new instance must use a fresh transport
    id. Before the unique-id fix this crashed on the second add with
    ExceedsMaxSupportedPublishers (the host still held the status publisher
    slot for the reused name)."""
    mgr = _bare_manager()
    try:
        params = {"common": {"process_group": "reuse_grp"}}
        for _ in range(5):
            n = mgr.add_node("Oscillator", "inputs", params=params)
            mgr.remove_node(n)
    finally:
        mgr.terminate(notify_gui=False)
