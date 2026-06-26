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
