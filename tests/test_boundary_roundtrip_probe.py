"""Throwaway probe: full authoring round-trip for In/Out boundary authoring."""
import os, uuid, atexit
import yaml

from goofi.manager import Manager, NodeContainer, _cleanup_iceoryx2_shm
from goofi.node_helpers import NodeProcessRegistry, list_nodes
from goofi.transport import set_instance_id


def _bare_manager() -> Manager:
    mgr = Manager.__new__(Manager)
    instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    set_instance_id(instance_id)
    atexit.register(_cleanup_iceoryx2_shm, instance_id)
    list_nodes(verbose=False)
    mgr._instance_id = instance_id
    mgr._headless = True
    mgr._use_multiprocessing = False
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


def test_full_authoring_roundtrip(tmp_path):
    # ---- author: add InArray, wire to a member, splice an external parent wire ----
    mgr = _bare_manager()
    osc = mgr.add_node("Oscillator", "inputs")
    buf = mgr.add_node("Buffer", "signal")
    inst = mgr.group_nodes([buf])  # inst::buffer0, empty interface
    bid = mgr.add_boundary(inst, "in", "ARRAY", pos=(11, 22))
    mgr.wire_boundary(inst, bid, "buffer0", "val")
    disp, slot = mgr.resolve_boundary(inst, bid)  # what bridge resolves to
    mgr.add_link(osc, disp, "out", slot)  # the spliced external (flat) link

    # sanity: the flat link exists, and the interface entry is wired
    flat = [dict(l) for l in mgr.links]
    assert {"node_out": osc, "node_in": disp, "slot_out": "out", "slot_in": slot} in flat, flat
    e = mgr._instances[inst]["interface"][bid]
    assert e["inner_node"] == "buffer0" and e["dtype"] == "ARRAY" and list(e["pos"]) == [11, 22]

    fp = str(tmp_path / "authored.gfi")
    mgr.save(fp, overwrite=True)

    doc = yaml.load(open(fp), Loader=yaml.FullLoader)
    print("=== SAVED INSTANCE INTERFACE ===")
    print(yaml.dump(doc["root"]["instances"][inst]["interface"]))
    print("=== SAVED ROOT LINKS ===")
    print(yaml.dump(doc["root"]["links"]))
    mgr.terminate()

    # ---- reload ----
    mgr2 = _bare_manager()
    mgr2.load(fp)
    links2 = [dict(l) for l in mgr2.links]
    print("=== RELOADED LINKS ===")
    for l in links2:
        print(l)
    print("=== RELOADED INTERFACE ===")
    print(mgr2._instances[inst]["interface"])

    # the flat external link restored
    assert {"node_out": osc, "node_in": disp, "slot_out": "out", "slot_in": slot} in links2, links2
    # the boundary entry restored, still wired, with dtype + pos
    e2 = mgr2._instances[inst]["interface"][bid]
    assert e2["inner_node"] == "buffer0" and e2["inner_slot"] == "val"
    assert e2["dtype"] == "ARRAY" and list(e2["pos"]) == [11, 22]
    mgr2.terminate()


def test_legacy_interface_only_loads(tmp_path):
    """A legacy v2 .gfi whose interface entries have NO dtype/pos and are always
    wired must still load (the frontend boundaryDtype falls back; backend reads)."""
    # Hand-craft a doc shaped like a pre-authoring interface entry.
    doc = {
        "version": 2,
        "definitions": {},
        "root": {
            "nodes": {
                "oscillator0": {
                    "_type": "Oscillator", "category": "inputs", "params": {},
                    "uid": "u-osc", "gui_kwargs": {"pos": [0, 0]},
                },
            },
            "links": [
                {"node_out": "oscillator0", "node_in": "sp0::buffer0", "slot_out": "out", "slot_in": "val"},
            ],
            "instances": {
                "sp0": {
                    "kind": "unique",
                    "pos": [100, 100],
                    "interface": {
                        # legacy shape: dir + inner only, no dtype/pos
                        "buffer0.val": {"dir": "in", "inner_node": "buffer0", "inner_slot": "val"},
                    },
                    "members": {
                        "buffer0": {
                            "_type": "Buffer", "category": "signal", "params": {},
                            "uid": "u-buf", "gui_kwargs": {"pos": [120, 120]},
                        },
                    },
                    "links": [],
                },
            },
        },
    }
    fp = str(tmp_path / "legacy.gfi")
    with open(fp, "w") as f:
        yaml.dump(doc, f, sort_keys=False)

    mgr = _bare_manager()
    mgr.load(fp)
    iface = mgr._instances["sp0"]["interface"]
    print("=== LEGACY LOADED INTERFACE ===", iface)
    assert iface["buffer0.val"]["inner_node"] == "buffer0"
    links = [dict(l) for l in mgr.links]
    print("=== LEGACY LINKS ===", links)
    assert {"node_out": "oscillator0", "node_in": "sp0::buffer0", "slot_out": "out", "slot_in": "val"} in links
    # Now re-save: does the legacy entry survive (no crash on missing dtype/pos)?
    fp2 = str(tmp_path / "legacy_resave.gfi")
    mgr.save(fp2, overwrite=True)
    doc2 = yaml.load(open(fp2), Loader=yaml.FullLoader)
    print("=== LEGACY RESAVED INTERFACE ===", doc2["root"]["instances"]["sp0"]["interface"])
    mgr.terminate()
