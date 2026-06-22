"""Atomic _transaction seam for multi-node sub-patch mutations (backlog #2, spec §2.10).

A failure partway through a multi-node op must leave the live graph byte-identical
to before: no orphan spawned process, and the in-memory state maps (_links,
_node_groups, _membership, _instances, _definitions) restored. This is the uniform
rollback primitive the splice ops route through.
"""
import pytest

from .test_manager import _bare_manager


def test_transaction_rolls_back_nodes_and_maps_on_failure():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        before_nodes = set(mgr.nodes)
        before_inst = dict(mgr._instances)
        with pytest.raises(RuntimeError):
            with mgr._transaction():
                mgr.add_node("Oscillator", "inputs")  # spawns a node (thread-mode)
                mgr._instances["ghost"] = {"kind": "unique", "members": {}}
                raise RuntimeError("boom")
        assert set(mgr.nodes) == before_nodes, "spawned node not torn down on rollback"
        assert mgr._instances == before_inst, "_instances not restored on rollback"
    finally:
        mgr.terminate(notify_gui=False)


def test_transaction_commits_on_success():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        before = set(mgr.nodes)
        with mgr._transaction():
            n = mgr.add_node("Oscillator", "inputs")
        assert n in mgr.nodes and set(mgr.nodes) == before | {n}
    finally:
        mgr.terminate(notify_gui=False)


def test_instantiate_definition_atomic_on_link_failure(monkeypatch):
    """A forced failure during the link phase tears down the just-spawned members
    and leaves no instance record behind (atomic via _transaction)."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst)
        before_nodes = set(mgr.nodes)
        before_inst = set(mgr._instances)

        calls = {"n": 0}
        real_add_node_from_record = mgr._add_node_from_record

        def boom(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 1:
                raise RuntimeError("forced spawn failure")
            return real_add_node_from_record(*args, **kwargs)

        monkeypatch.setattr(mgr, "_add_node_from_record", boom)
        with pytest.raises(RuntimeError):
            mgr.instantiate_definition(def_id)

        assert set(mgr.nodes) == before_nodes, "orphan member left after failed instantiate"
        assert set(mgr._instances) == before_inst, "instance record left after failed instantiate"
    finally:
        mgr.terminate(notify_gui=False)
