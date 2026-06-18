"""Tests for the Manager after the iceoryx2 refactor."""
import time
import uuid
from os import path

import pytest

import goofi
from goofi.manager import Manager, NodeContainer
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
    """Oscillator → Select with a 0:5 include filter."""
    osc = mgr.add_node("Oscillator", "inputs")
    sel = mgr.add_node("Select", "array", params={"select": {"include": "0:5"}})
    mgr.add_link(osc, sel, "out", "data")
    return osc, sel


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
        uids_before = {name: mgr.nodes[name].member_uid for name in (osc, sel)}
        assert all(uids_before.values())
        fp = str(tmp_path / "p.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        uids_after = {name: mgr2.nodes[name].member_uid for name in (osc, sel)}
        assert uids_after == uids_before  # uid is stable across save/load
    finally:
        mgr2.terminate()


def test_load_v1_flat_patch_still_works(tmp_path):
    # A legacy flat patch (no `version` key) must still load and get fresh uids.
    import yaml as _yaml
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        _build_simple_graph(mgr)
        v2_text = mgr.serialize_patch()
        doc = _yaml.load(v2_text, Loader=_yaml.FullLoader)
    finally:
        mgr.terminate()
    # Strip to the legacy flat shape (drop uids too).
    flat = {"nodes": {}, "links": doc["root"]["links"]}
    for name, n in doc["root"]["nodes"].items():
        n.pop("uid", None)
        flat["nodes"][name] = n
    fp = tmp_path / "legacy.gfi"
    fp.write_text(_yaml.dump(flat, sort_keys=False))

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(str(fp))
        assert len(mgr2.nodes) == 2
        # fresh uids minted on load
        assert all(mgr2.nodes[name].member_uid for name in mgr2.nodes)
    finally:
        mgr2.terminate()


def _build_grouped_graph(mgr):
    """osc → sel0 → sel1, with [sel0, sel1] grouped into one sub-patch.

    Returns (osc_name, inst_id, members) where members maps display name -> local.
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
        assert inst == "subpatch0"
        assert "subpatch0::select0" in mgr.nodes
        assert "subpatch0::select1" in mgr.nodes
        assert "select0" not in mgr.nodes  # renamed in place
        assert mgr._membership["subpatch0::select0"] == "subpatch0"
        # external link followed the rename
        assert {"node_out": osc, "node_in": "subpatch0::select0", "slot_out": "out", "slot_in": "data"} in [
            dict(link) for link in mgr.links
        ]
        # interface derived: select0.data is a boundary input (fed by osc)
        assert "select0.data" in mgr._instances[inst]["interface"]
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
        assert "subpatch0" in doc["root"]["instances"]
        assert set(doc["root"]["instances"]["subpatch0"]["members"]) == {"select0", "select1"}
        assert len(doc["root"]["instances"]["subpatch0"]["links"]) == 1  # internal sel0->sel1
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        assert "subpatch0::select0" in mgr2.nodes and "subpatch0::select1" in mgr2.nodes
        assert mgr2._membership["subpatch0::select1"] == "subpatch0"
        assert set(mgr2._instances["subpatch0"]["members"]) == {
            "subpatch0::select0",
            "subpatch0::select1",
        }
        # both the external and internal links are restored
        links = [dict(link) for link in mgr2.links]
        assert {"node_out": osc, "node_in": "subpatch0::select0", "slot_out": "out", "slot_in": "data"} in links
        assert any(l["node_out"] == "subpatch0::select0" and l["node_in"] == "subpatch0::select1" for l in links)

        # expand dissolves the group back to bare names
        restored = mgr2.expand_instance("subpatch0")
        assert set(restored) == {"select0", "select1"}
        assert "subpatch0::select0" not in mgr2.nodes
        assert "select0" in mgr2.nodes
        assert mgr2._instances == {} and mgr2._membership == {}
        links2 = [dict(link) for link in mgr2.links]
        assert {"node_out": osc, "node_in": "select0", "slot_out": "out", "slot_in": "data"} in links2
    finally:
        mgr2.terminate()


def test_shared_param_edit_mirrors_to_sibling():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # unique sub-patch with select0, select1
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        # both instances reference the same definition
        assert mgr._instances[inst]["def_id"] == def_id
        assert mgr._instances[inst2]["def_id"] == def_id
        # edit a param on instance A's select0 → mirrors to instance B's select0
        mgr.update_param(f"{inst}::select0", "select", "include", "3:9")
        assert mgr.nodes[f"{inst2}::select0"].params["select"]["include"].value == "3:9"
        # the definition (save source of truth) also reflects the edit
        assert mgr._definitions[def_id]["members"]["select0"]["params"]["select"]["include"] == "3:9"
    finally:
        mgr.terminate()


def test_make_unique_detaches_and_gcs_definition():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.make_unique(inst2)
        assert mgr._instances[inst2]["def_id"] is None
        assert mgr._instances[inst2]["kind"] == "unique"
        # def still referenced by `inst`
        assert def_id in mgr._definitions
        # editing inst2 no longer affects inst
        mgr.update_param(f"{inst2}::select0", "select", "include", "7:8")
        assert mgr.nodes[f"{inst}::select0"].params["select"]["include"].value != "7:8"
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
        assert mgr._instances[inst]["interface"] == {}
        bid = mgr.add_boundary(inst, "in", "ARRAY", pos=(5, 6))
        e = mgr._instances[inst]["interface"][bid]
        assert e == {"dir": "in", "dtype": "ARRAY", "inner_node": None, "inner_slot": None, "pos": [5, 6]}
        mgr.wire_boundary(inst, bid, "buffer0", "val")
        e = mgr._instances[inst]["interface"][bid]
        assert e["inner_node"] == "buffer0" and e["inner_slot"] == "val"
        assert mgr.resolve_boundary(inst, bid) == (f"{inst}::buffer0", "val")
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
        mgr.remove_node(f"{inst}::buffer1")
        # Member dropped from the instance + membership; boundary unwired (not dangling).
        assert f"{inst}::buffer1" not in mgr._instances[inst]["members"]
        assert f"{inst}::buffer1" not in mgr._membership
        assert mgr._instances[inst]["interface"][bid]["inner_node"] is None
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
            mgr.remove_node(f"{inst}::buffer0")  # shared member — must make unique first
    finally:
        mgr.terminate()


def test_wire_legacy_entry_without_dtype_heals():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        # A legacy interface entry (pre-dtype shape) must not crash on wire.
        mgr._instances[inst]["interface"]["legacy0"] = {
            "dir": "in",
            "inner_node": None,
            "inner_slot": None,
            "pos": [0, 0],
        }
        mgr.wire_boundary(inst, "legacy0", "buffer0", "val")
        assert mgr._instances[inst]["interface"]["legacy0"]["dtype"] == "ARRAY"
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
        disp, slot = mgr.resolve_boundary(inst, bid)  # what the bridge resolves to
        mgr.add_link(osc, disp, "out", slot)          # the spliced flat link
        assert any(l["node_out"] == osc and l["node_in"] == disp and l["slot_in"] == slot for l in mgr.links)
        # Deleting the boundary tears down its external wire.
        mgr.remove_boundary(inst, bid)
        assert bid not in mgr._instances[inst]["interface"]
        assert not any(l["node_in"] == disp and l["slot_in"] == slot for l in mgr.links)
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
        mgr.add_link(osc, f"{inst}::buffer0", "out", "val")  # external splice to buffer0
        # Re-wire the In node to buffer1 → its external link follows.
        mgr.wire_boundary(inst, bid, "buffer1", "val")
        assert any(l["node_out"] == osc and l["node_in"] == f"{inst}::buffer1" and l["slot_in"] == "val" for l in mgr.links)
        assert not any(l["node_in"] == f"{inst}::buffer0" and l["node_out"] == osc for l in mgr.links)
    finally:
        mgr.terminate()


def test_shared_boundary_mirrors_and_make_unique_isolates():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        buf, inst = _build_single_member_subpatch(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        bid = mgr.add_boundary(inst, "in", "ARRAY")  # mirrors to def + sibling
        assert bid in mgr._instances[inst2]["interface"]
        assert bid in mgr._definitions[def_id]["interface"]
        mgr.wire_boundary(inst, bid, "buffer0", "val")  # inner mirrors to sibling's own member
        assert mgr._instances[inst2]["interface"][bid]["inner_node"] == "buffer0"
        # Detach inst2; editing it must not cross-mutate inst (deep-copied interface).
        mgr.make_unique(inst2)
        mgr.wire_boundary(inst2, bid, None, None)
        assert mgr._instances[inst2]["interface"][bid]["inner_node"] is None
        assert mgr._instances[inst]["interface"][bid]["inner_node"] == "buffer0"
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
        e = mgr2._instances[inst]["interface"][bid]
        assert e["inner_node"] is None and e["dtype"] == "ARRAY" and list(e["pos"]) == [7, 8]
    finally:
        mgr2.terminate()


def test_shared_member_pos_mirrors_to_sibling():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        # Move a member inside instance A → mirrors to B's corresponding member.
        changed = mgr.set_node_pos(f"{inst}::select0", (123, 456))
        assert list(mgr.nodes[f"{inst}::select0"].gui_kwargs["pos"]) == [123, 456]
        assert list(mgr.nodes[f"{inst2}::select0"].gui_kwargs["pos"]) == [123, 456]
        # Definition (save source of truth) records the pos too.
        assert list(mgr._definitions[def_id]["members"]["select0"]["gui_kwargs"]["pos"]) == [123, 456]
        # Both the moved node and its sibling are reported changed (for broadcasts).
        assert set(changed) == {f"{inst}::select0", f"{inst2}::select0"}
        # A non-mirrored member (select1) is untouched in the sibling.
        assert mgr.nodes[f"{inst2}::select1"].gui_kwargs.get("pos") != [123, 456]
    finally:
        mgr.terminate()


def test_remove_instance_deletes_members_and_gcs_def():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        # Delete one shared instance → its members + links gone; def survives (inst2 holds it).
        mgr.remove_instance(inst)
        assert inst not in mgr._instances
        assert f"{inst}::select0" not in mgr.nodes
        assert f"{inst}::select0" not in mgr._membership
        assert all(link["node_in"] != f"{inst}::select0" for link in mgr.links)
        assert def_id in mgr._definitions
        # Delete the last instance → orphaned definition is GC'd.
        mgr.remove_instance(inst2)
        assert def_id not in mgr._definitions
        assert mgr._instances == {}
        assert all(not k.startswith(f"{inst2}::") for k in mgr._membership)
    finally:
        mgr.terminate()


def test_shared_member_pos_survives_save_load(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.set_node_pos(f"{inst}::select0", (321, 654))
        fp = str(tmp_path / "shared_pos.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate()

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        # The mirrored layout round-trips: every shared sibling's member is at the moved pos.
        shared_ids = [iid for iid, i in mgr2._instances.items() if i["def_id"] == def_id]
        assert len(shared_ids) == 2
        for iid in shared_ids:
            assert list(mgr2.nodes[f"{iid}::select0"].gui_kwargs["pos"]) == [321, 654]
    finally:
        mgr2.terminate()


def test_unique_member_pos_does_not_mirror():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)  # unique sub-patch
        changed = mgr.set_node_pos(f"{inst}::select0", (10, 20))
        assert list(mgr.nodes[f"{inst}::select0"].gui_kwargs["pos"]) == [10, 20]
        assert changed == [f"{inst}::select0"]
    finally:
        mgr.terminate()


def test_make_unique_stops_pos_mirror():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        osc, inst = _build_grouped_graph(mgr)
        def_id = mgr.share_instance(inst)
        inst2 = mgr.instantiate_definition(def_id)
        mgr.make_unique(inst2)
        changed = mgr.set_node_pos(f"{inst}::select0", (7, 8))
        # inst2 detached → its member must not move with inst's.
        assert mgr.nodes[f"{inst2}::select0"].gui_kwargs.get("pos") != [7, 8]
        assert changed == [f"{inst}::select0"]
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
        shared_ids = [iid for iid, i in mgr2._instances.items() if i["def_id"] == def_id]
        assert len(shared_ids) == 2
        a, b = shared_ids
        mgr2.update_param(f"{a}::select0", "select", "include", "1:4")
        assert mgr2.nodes[f"{b}::select0"].params["select"]["include"].value == "1:4"
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
        mgr.remove_node(n1)
        n2 = mgr.add_node("Oscillator", "inputs")
        id2 = mgr.nodes[n2].node_id
        # The display name is reused (nice UX); the transport id is not.
        assert n1 == n2 == "oscillator0"
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
        directory = mgr._node_directory()
        assert directory[a] == mgr.nodes[a].node_id
        assert directory[b] == mgr.nodes[b].node_id
        assert directory[a] != directory[b]
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
