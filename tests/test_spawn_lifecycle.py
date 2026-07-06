"""Child-side import: the manager process never imports the node module; a
bootstrap failure is a permanent, observable error — not a restart loop."""

import dataclasses
import sys
import time

import pytest

from goofi.node import spawn_node
from goofi.registry import build_catalog

CATALOG, _ = build_catalog()


def wait_until(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_spawning_process_does_not_import_node_module():
    spec = CATALOG["ConstantArray"]
    if spec.module in sys.modules:
        pytest.skip("node module already imported by an earlier test in this process")
    ref = spawn_node(spec, initial_params={"common": {"autotrigger": False}})
    try:
        assert spec.module not in sys.modules  # the import happened in the child only
        assert ref.stage == "creating"
        assert ref.wait_for_state(timeout=10.0)
        assert wait_until(lambda: ref.stage in ("setup", "ready"))
        assert spec.module not in sys.modules
    finally:
        ref.terminate()


def test_bootstrap_import_error_reports_over_the_pipe():
    spec = dataclasses.replace(CATALOG["ConstantArray"], module="goofi.nodes.nope.missing")
    ref = spawn_node(spec)
    try:
        assert wait_until(lambda: ref.process is not None and not ref.process.is_alive())
        assert ref.process.exitcode != 0
        assert ref.boot_conn is not None and ref.boot_conn.poll(1.0)
        tb = ref.boot_conn.recv()
        assert "ModuleNotFoundError" in tb
        assert ref.stage == "creating"  # never came up; the supervisor flips it to error
    finally:
        ref.terminate()


def test_supervisor_marks_boot_failure_permanent_no_restart():
    """A node whose process dies before its first STATE_UPDATE failed bootstrap:
    the sweep must surface the traceback as a terminal error stage and must NOT
    respawn it (restarting would loop forever on the same traceback)."""
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        spec = dataclasses.replace(mgr.node_specs["ConstantArray"], module="goofi.nopes.missing")
        ref = spawn_node(spec)
        ref.uid = "boot-fail-uid"
        ref.name = "bootfail0"
        mgr.nodes.add_node(ref.uid, ref)
        assert wait_until(lambda: not ref.process.is_alive())

        mgr._supervise_once()

        assert mgr.nodes[ref.uid] is ref  # not replaced by a respawn
        assert ref.stage == "error"
        assert "ModuleNotFoundError" in (ref.last_error or "")
    finally:
        mgr.terminate(notify_gui=False)


def test_crash_after_ready_still_restarts():
    """The existing crash-restart policy is untouched for nodes that WERE healthy."""
    from goofi.manager import Manager

    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        uid = mgr.add_node("ConstantArray", "inputs", params={"common": {"autotrigger": False}})
        ref = mgr.nodes[uid]
        assert ref.wait_for_state(timeout=10.0)
        assert Manager._should_restart(ref) is True

        ref.process.kill()
        assert wait_until(lambda: not ref.process.is_alive())
        mgr._supervise_once()

        new_ref = mgr.nodes[uid]
        assert new_ref is not ref  # respawned
        assert new_ref.restart_count == 1
    finally:
        mgr.terminate(notify_gui=False)


def test_add_node_returns_before_ready():
    """add_node must not block on the child's first state push — the node is
    tracked (and broadcast) from the moment of add."""
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        uid = mgr.add_node("ConstantArray", "inputs", params={"common": {"autotrigger": False}})
        ref = mgr.nodes[uid]
        assert ref.stage == "creating"  # returned before the first STATE_UPDATE
        assert wait_until(lambda: ref.stage == "ready")
    finally:
        mgr.terminate(notify_gui=False)


def test_ctrl_messages_queue_until_ready():
    """add + link + param issued before the child is up must all apply, in order."""
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        a = mgr.add_node("ConstantArray", "inputs", params={"common": {"autotrigger": False}})
        b = mgr.add_node("Math", "array")
        mgr.add_link(a, b, "out", "data")
        mgr.nodes[b].update_param("math", "multiply", 3.0)
        assert wait_until(lambda: mgr.nodes[a].stage == "ready" and mgr.nodes[b].stage == "ready")
        # the queued link materialized: the upstream slot gained a subscriber
        assert wait_until(
            lambda: (mgr.nodes[a].serialized_state or {})
            .get("output_subscribers", {})
            .get("out", 0)
            >= 1
        )
        # the queued param update reached the node (echoed in its state push)
        assert wait_until(
            lambda: (mgr.nodes[b].serialized_state or {}).get("params", {}).get("math", {}).get("multiply")
            == 3.0
        )
    finally:
        mgr.terminate(notify_gui=False)


def test_stage_passes_through_setup_to_ready():
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        uid = mgr.add_node("ConstantArray", "inputs", params={"common": {"autotrigger": False}})
        ref = mgr.nodes[uid]
        seen = set()
        assert wait_until(lambda: (seen.add(ref.stage), ref.stage == "ready")[1])
        assert seen <= {"creating", "setup", "ready"} and "ready" in seen

        from goofi.bridge.schemas import describe_node_instance

        assert describe_node_instance(uid, ref)["stage"] == "ready"
    finally:
        mgr.terminate(notify_gui=False)


def test_terminate_while_creating_kills_the_process():
    """A TERMINATE publish can't reach a child that has no endpoints yet —
    tearing down a never-ready node must kill the process directly."""
    spec = dataclasses.replace(CATALOG["ConstantArray"], module="tests._slow_import", cls_name="SlowNode")
    ref = spawn_node(spec)
    try:
        assert ref.stage == "creating"
        t0 = time.time()
        ref.terminate()
        assert wait_until(lambda: not ref.process.is_alive(), timeout=2.0)
        assert time.time() - t0 < 2.0
    finally:
        ref.terminate()


def test_child_reported_options_reach_the_ref():
    """The node's state push carries its StringParam option lists; the ref merges
    them so dynamic device lists refresh from the live process (and the bridge's
    state_update rebroadcast carries them to dropdowns for free)."""
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        uid = mgr.add_node("ConstantArray", "inputs", params={"common": {"autotrigger": False}})
        ref = mgr.nodes[uid]
        assert wait_until(lambda: ref.stage == "ready")
        assert "param_options" in (ref.serialized_state or {})
    finally:
        mgr.terminate(notify_gui=False)


def test_burst_add_all_reach_ready():
    """Regression: spawn_node must open the manager-side ref (status
    subscriber) BEFORE starting the child. iceoryx2 keeps no history, so a
    subscriber opened after the child's first STATE_UPDATE loses it — and the
    pre-ready ctrl queue then withholds every message, so the node (never
    dirtied again) deadlocks in 'creating'. A 10-node burst reproduced this
    reliably."""
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        uids = [mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": False}})]
        uids.append(mgr.add_node("PSD", "signal"))
        for _ in range(8):
            uids.append(mgr.add_node("Buffer", "signal"))
        assert wait_until(
            lambda: all(mgr.nodes[u].stage == "ready" for u in uids), timeout=15.0
        ), f"stuck stages: { {u: mgr.nodes[u].stage for u in uids if mgr.nodes[u].stage != 'ready'} }"
    finally:
        mgr.terminate(notify_gui=False)
