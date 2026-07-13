"""Phase 4 — process-level GC/scheduling policy (`_apply_gc_policy`)."""
import gc
import os
import sys

import pytest

from goofi.node import NodeEnv

from .utils import DummyNode


def _mp_node():
	"""A node that OWNS its process — the only environment where mutating
	process-global GC/scheduling state is safe."""
	node = DummyNode.create_standalone()
	node._environment = NodeEnv.MULTIPROCESSING
	return node


def test_apply_gc_policy_normal_freezes_once(monkeypatch):
	calls = {"freeze": 0, "disable": 0}
	switch = []
	monkeypatch.setattr(gc, "freeze", lambda: calls.__setitem__("freeze", calls["freeze"] + 1))
	monkeypatch.setattr(gc, "disable", lambda: calls.__setitem__("disable", calls["disable"] + 1))
	monkeypatch.setattr(sys, "setswitchinterval", lambda v: switch.append(v))

	node = _mp_node()
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

	node = _mp_node()
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

	node = _mp_node()
	node.params.common.priority.value = "realtime"
	node._apply_gc_policy()  # must not raise

	assert node._gc_policy_applied is True


@pytest.mark.parametrize("env", [NodeEnv.LOCAL, NodeEnv.STANDALONE])
def test_gc_policy_skipped_for_colocated_nodes(monkeypatch, env):
	# In LOCAL (--no-multiprocessing or a shared process_group) and STANDALONE the
	# node does NOT own its process, so freezing/disabling GC or grabbing SCHED_FIFO
	# would corrupt the manager/bridge or sibling nodes. The policy must no-op there
	# — but still latch so the loop stops retrying it every tick.
	touched = []
	monkeypatch.setattr(gc, "freeze", lambda: touched.append("freeze"))
	monkeypatch.setattr(gc, "disable", lambda: touched.append("disable"))
	monkeypatch.setattr(sys, "setswitchinterval", lambda v: touched.append("switch"))

	node = DummyNode.create_standalone()
	node._environment = env
	node.params.common.priority.value = "realtime"
	node._apply_gc_policy()

	assert touched == []  # no process-global mutation in a co-located process
	assert node._gc_policy_applied is True  # latched: the loop won't retry


def test_gc_policy_applied_after_first_tick():
	"""The loop must invoke `_apply_gc_policy` once a real tick succeeds — after
	warm-up, never blocking setup. Driven with a live local node (autotrigger
	free-runs the loop) so this exercises the actual wiring, not the method.

	`create_local` runs the node in the LOCAL environment, so the policy latches
	but its process-global mutations are gated off (this is the manager's process)
	— nothing leaks into the rest of the suite."""
	import time
	import uuid

	from goofi.transport import set_instance_id

	from .utils import DummyNode

	set_instance_id(f"t-{uuid.uuid4().hex[:8]}")
	ref, n = DummyNode.create_local(initial_params={"common": {"autotrigger": True}})
	try:
		deadline = time.time() + 3.0
		while not n._gc_policy_applied and time.time() < deadline:
			time.sleep(0.01)
		assert n._gc_policy_applied, "GC policy was not applied after the first tick."
	finally:
		ref.terminate()
