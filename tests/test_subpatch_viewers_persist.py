"""Sub-patch instance viewer-state persists into the v2 envelope (backlog #17).

A collapsed sub-patch instance's per-boundary-slot viewer state (kind/settings) was
accept-and-ignored, so it was dropped on reload. It now rides on the instance record
and round-trips through build_v2_tree / _expand_doc.
"""
from .test_manager import _bare_manager


def test_instance_viewers_serialize_into_v2_tree():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        mgr._instances[inst]["viewers"] = {"out0": {"kind": "image", "settings": {}}}

        _root_nodes, _root_links, _defs, instances = mgr.build_v2_tree()
        assert instances[inst].get("viewers") == {"out0": {"kind": "image", "settings": {}}}
    finally:
        mgr.terminate(notify_gui=False)


def test_instance_viewers_restore_on_expand_doc():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        mgr._instances[inst]["viewers"] = {"out0": {"kind": "topomap"}}
        root_nodes, root_links, defs, instances = mgr.build_v2_tree()

        # Reload into a fresh manager via the same expand path the loader uses.
        mgr2 = _bare_manager(use_multiprocessing=False)
        try:
            mgr2._expand_doc(root_nodes, root_links, instances, defs)
            assert mgr2._instances[inst].get("viewers") == {"out0": {"kind": "topomap"}}
        finally:
            mgr2.terminate(notify_gui=False)
    finally:
        mgr.terminate(notify_gui=False)
