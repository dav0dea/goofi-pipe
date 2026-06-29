"""Atomic _transaction seam for multi-node sub-patch mutations (backlog #2, spec §2.10).

A failure partway through a multi-node op must leave the live graph byte-identical
to before: no orphan spawned process, and the in-memory state maps (_links,
_node_groups, _membership, _instances, _definitions) restored. This is the uniform
rollback primitive the splice ops route through.
"""
from copy import deepcopy

import pytest

from .test_manager import _bare_manager


def test_transaction_rolls_back_nodes_and_maps_on_failure():
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        before_nodes = set(mgr.nodes)
        # deepcopy: a top-level add now mutates ROOT.members in place, so a shallow copy
        # would alias the live (mutated-then-rolled-back) ROOT record.
        before_inst = deepcopy(mgr._instances)
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


def test_load_missing_definition_aborts_clean_without_leaking_nodes():
    """A doc whose shared instance references an absent definition must fail fast,
    before spawning anything — not spawn the root graph and then leak it when the
    missing def aborts the splice mid-way."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        # a VALID root-node record (so its spawn succeeds — the leak we're guarding
        # against is this node surviving when the later missing-def aborts the splice)
        osc = mgr.add_node("Oscillator", "inputs")
        rec = mgr._node_record(osc)
        mgr.remove_node(osc)
        assert len(mgr.nodes) == 0
        root_nodes = {rec["uid"]: rec}
        instances = {
            "subpatch0": {
                "kind": "shared",
                "def": "missing",
                "pos": [0, 0],
                "members": {"x0": {"uid": "deaduid", "pos": [0, 0]}},
            }
        }
        with pytest.raises(KeyError):
            mgr._expand_doc(root_nodes, [], instances, definitions={})
        assert len(mgr.nodes) == 0, "partial graph leaked after a missing-definition abort"
    finally:
        mgr.terminate(notify_gui=False)


def test_add_member_node_atomic_on_sibling_spawn_failure(monkeypatch):
    """Adding a member to a SHARED family mirrors the spawn into the def + every
    sibling. If a sibling spawn fails mid-loop, the whole add must roll back — not
    leave the primary + def + some siblings carrying a member the rest lack."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        inst = mgr.group_nodes([a])
        def_id = mgr.share_instance(inst)
        mgr.instantiate_definition(def_id)  # sibling B
        mgr.instantiate_definition(def_id)  # sibling C

        before_nodes = set(mgr.nodes)
        before_def = set(mgr._definitions[def_id].members)
        before_members = {iid: dict(mgr._instances[iid].members) for iid in mgr._instances}

        real = mgr._add_node_from_record
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 1:  # fail the FIRST sibling spawn
                raise RuntimeError("sibling spawn failed")
            return real(*args, **kwargs)

        monkeypatch.setattr(mgr, "_add_node_from_record", boom)
        with pytest.raises(RuntimeError):
            mgr.add_member_node(inst, "Buffer", "signal")

        assert set(mgr.nodes) == before_nodes, "leaked member node after failed add_member"
        assert set(mgr._definitions[def_id].members) == before_def, "definition half-mirrored"
        assert {iid: dict(mgr._instances[iid].members) for iid in mgr._instances} == before_members, "instances half-mirrored"
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


def test_group_reparent_rolls_back_on_failure(monkeypatch):
    """A reparenting group (Phase 3a) detaches members from one parent and re-homes
    them under a new nested child — a multi-scope mutation. A failure partway must
    restore _membership and _instances byte-clean, proving no parent-edge state
    (the now-live SubPatchInstance.parent) escapes the existing _transaction snapshot."""
    from copy import deepcopy

    mgr = _bare_manager(use_multiprocessing=False)
    try:
        a = mgr.add_node("Buffer", "signal")
        b = mgr.add_node("Buffer", "signal")
        c = mgr.add_node("Buffer", "signal")
        P = mgr.group_nodes([a, b, c])
        before_membership = dict(mgr._membership)
        before_instances = deepcopy(mgr._instances)

        real_attach = mgr._attach_member
        calls = {"n": 0}

        def boom(inst_id, uid, local):
            calls["n"] += 1
            if calls["n"] == 1:  # fail on the first re-home, after members were detached
                raise RuntimeError("forced reparent failure")
            return real_attach(inst_id, uid, local)

        monkeypatch.setattr(mgr, "_attach_member", boom)
        with pytest.raises(RuntimeError):
            mgr.group_nodes([a, b])

        assert mgr._membership == before_membership, "_membership not restored on rollback"
        assert set(mgr._instances) == set(before_instances), "stray instance left after rollback"
        assert set(mgr._instances[P].members) == set(before_instances[P].members), (
            "parent's members not restored (a/b not returned to P) on rollback"
        )
        assert all(mgr._instances[i].parent == before_instances[i].parent for i in before_instances), (
            ".parent markers not restored on rollback"
        )
    finally:
        mgr.terminate(notify_gui=False)


def test_transaction_rolls_back_unsaved_changes_flag_on_failure():
    """A rolled-back transaction must leave the manager BYTE-IDENTICAL — including the
    unsaved_changes flag. An inner @mark_unsaved_changes call (e.g. wire_boundary inside
    the shared mirror-remove) sets the flag before a LATER op in the same transaction
    fails; the rollback restores the state maps, so it must restore this flag too, or the
    UI shows false 'unsaved changes' after an op that changed nothing."""
    mgr = _bare_manager(use_multiprocessing=False)
    try:
        mgr.unsaved_changes = False
        with pytest.raises(RuntimeError):
            with mgr._transaction():
                mgr.unsaved_changes = True  # an inner decorated call's side effect
                raise RuntimeError("boom")
        assert mgr.unsaved_changes is False  # restored to the pre-transaction value
    finally:
        mgr.terminate(notify_gui=False)
