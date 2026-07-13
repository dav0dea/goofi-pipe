"""Phase 4 — process-level GC/scheduling policy (`_apply_gc_policy`)."""
import gc
import os
import sys

from .utils import DummyNode


def test_apply_gc_policy_normal_freezes_once(monkeypatch):
	calls = {"freeze": 0, "disable": 0}
	switch = []
	monkeypatch.setattr(gc, "freeze", lambda: calls.__setitem__("freeze", calls["freeze"] + 1))
	monkeypatch.setattr(gc, "disable", lambda: calls.__setitem__("disable", calls["disable"] + 1))
	monkeypatch.setattr(sys, "setswitchinterval", lambda v: switch.append(v))

	node = DummyNode.create_standalone()
	assert node._gc_policy_applied is False
	node._apply_gc_policy()
	node._apply_gc_policy()  # idempotent: guard must swallow the second call

	assert calls["freeze"] == 1
	assert calls["disable"] == 0  # normal priority never disables GC
	assert switch == [0.001]
	assert node._gc_policy_applied is True


def test_apply_gc_policy_realtime_disables_and_sets_fifo(monkeypatch):
	calls = {"disable": 0}
	sched = []
	monkeypatch.setattr(gc, "freeze", lambda: None)
	monkeypatch.setattr(gc, "disable", lambda: calls.__setitem__("disable", calls["disable"] + 1))
	monkeypatch.setattr(sys, "setswitchinterval", lambda v: None)
	monkeypatch.setattr(os, "sched_setscheduler", lambda pid, policy, param: sched.append((pid, policy)))

	node = DummyNode.create_standalone()
	node.params.common.priority.value = "realtime"
	node._apply_gc_policy()

	assert calls["disable"] == 1
	assert sched and sched[0][0] == 0  # applied to this process


def test_apply_gc_policy_realtime_permission_denied_degrades(monkeypatch):
	monkeypatch.setattr(gc, "freeze", lambda: None)
	monkeypatch.setattr(gc, "disable", lambda: None)
	monkeypatch.setattr(sys, "setswitchinterval", lambda v: None)

	def boom(*a, **k):
		raise PermissionError("no rtprio")

	monkeypatch.setattr(os, "sched_setscheduler", boom)

	node = DummyNode.create_standalone()
	node.params.common.priority.value = "realtime"
	node._apply_gc_policy()  # must not raise

	assert node._gc_policy_applied is True
