"""Stable instance identity (Phase 2b of the recursive-sub-patch redesign).

A sub-patch instance is a first-class, uid-identified entity — exactly like a node:
its `_instances` key is a STABLE minted uid (never the reused `subpatch0` label),
and its display name is a separate, auto-numbered field. The uid persists across
save/load, so the frontend can reconcile instances by a stable key (no more
"subpatch0 regenerated -> stale link silently re-binds").
"""
import string

from goofi.manager import ROOT_ID

from .test_manager import _bare_manager


def _is_uid(s: str) -> bool:
    return len(s) == 12 and all(c in string.hexdigits for c in s)


def test_instance_uid_is_opaque_and_name_is_separate():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        # the key is a minted uid, not the display label
        assert _is_uid(inst)
        assert mgr._instances[inst].uid == inst
        assert mgr._instances[inst].name == "subpatch0"
        assert inst != mgr._instances[inst].name
    finally:
        mgr.terminate(notify_gui=False)


def test_instance_display_names_are_unique_and_auto_numbered():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        i1 = mgr.group_nodes([a])
        i2 = mgr.group_nodes([b])
        assert {mgr._instances[i1].name, mgr._instances[i2].name} == {"subpatch0", "subpatch1"}
        assert i1 != i2
    finally:
        mgr.terminate(notify_gui=False)


def test_instance_uid_persists_across_save_load(tmp_path):
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        b = mgr.add_node("Buffer", "signal")
        i1 = mgr.group_nodes([a])
        def_id = mgr.share_instance(i1)
        i2 = mgr.instantiate_definition(def_id)
        before = {i1: mgr._instances[i1].name, i2: mgr._instances[i2].name}
        fp = str(tmp_path / "ids.gfi")
        mgr.save(fp, overwrite=True)
    finally:
        mgr.terminate(notify_gui=False)

    mgr2 = _bare_manager(use_multiprocessing=False)
    try:
        mgr2.load(fp)
        # the same instance uids (and their names) come back (ROOT aside)
        assert set(mgr2._instances) - {ROOT_ID} == set(before)
        for iid, name in before.items():
            assert mgr2._instances[iid].uid == iid
            assert mgr2._instances[iid].name == name
    finally:
        mgr2.terminate(notify_gui=False)


def test_instance_uid_never_collides_with_a_node_uid():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        nodes = [mgr.add_node("Oscillator", "inputs") for _ in range(5)]
        insts = [mgr.group_nodes([u]) for u in nodes]
        assert set(insts).isdisjoint(set(mgr.nodes))
    finally:
        mgr.terminate(notify_gui=False)
