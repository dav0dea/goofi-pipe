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
