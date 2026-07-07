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


def test_spawn_feeds_child_raw_params_not_ast_fallback(tmp_path):
    """spawn_node must hand the child the caller's raw overrides, not the
    manager-side AST-fallback params. A dynamic node's static catalog default is
    a placeholder ("fallback"); force-feeding it into the child clobbers the
    child's real _configure() default ("real-a") — the AudioStream device bug."""
    import dataclasses
    import pathlib

    from .test_registry import make_nodes_tree

    src = (pathlib.Path(__file__).parent / "_divergent_node.py").read_text()
    ast_cat, errors = build_catalog(make_nodes_tree(tmp_path, {"misc/divergent.py": src}), package="fixture.nodes")
    ast_spec = ast_cat["DivergentDefault"]
    assert ast_spec.dynamic is True
    _, _, ast_params = ast_spec.configure()
    assert ast_params["g"]["choice"].value == "fallback"  # static catalog sees only the fallback

    # Point the spec at the real, importable module (child runs the real
    # _configure -> "real-a") while keeping the AST fallback hooks.
    spec = dataclasses.replace(ast_spec, module="tests._divergent_node", cls_name="DivergentDefault")
    ref = spawn_node(spec, initial_params=None)
    try:
        assert ref.wait_for_state(timeout=10.0)
        got = (ref.serialized_state or {}).get("params", {}).get("g", {}).get("choice")
        assert got == "real-a", f"child param clobbered by AST fallback: {got!r}"
    finally:
        ref.terminate()


def test_setup_failure_surfaces_error_via_state_plane():
    """A node whose setup() raises must surface the error, not hang on an eternal
    'setting up…' spinner. Its first PROCESSING_ERROR is lost (iceoryx2 keeps no
    history), so the error rides the idempotent STATE_UPDATE and reaches the ref."""
    spec = dataclasses.replace(CATALOG["ConstantArray"], module="tests._setup_fail", cls_name="SetupFail")
    ref = spawn_node(spec)
    try:
        assert ref.wait_for_state(timeout=10.0)  # the STATE_UPDATE (setup_complete=False) arrives
        assert wait_until(lambda: ref.last_error is not None and "setup boom" in ref.last_error), (
            "setup() error never surfaced on the ref"
        )
        assert ref.stage == "setup"  # setup failed, so it never reaches 'ready'
    finally:
        ref.terminate()


def test_setup_error_fans_out_to_the_processing_error_callback():
    """The carried state-plane error must drive the SAME fan-out as a live
    PROCESSING_ERROR (so the bridge broadcasts it + ancestor instances update),
    since the node's first PROCESSING_ERROR message was lost."""
    from goofi.message import MessageType

    seen = []
    spec = dataclasses.replace(CATALOG["ConstantArray"], module="tests._setup_fail", cls_name="SetupFail")
    ref = spawn_node(spec)
    ref.set_message_handler(
        MessageType.PROCESSING_ERROR, lambda nr, msg: seen.append(msg.content.get("error"))
    )
    try:
        assert wait_until(lambda: any(e and "setup boom" in e for e in seen), timeout=10.0)
    finally:
        ref.terminate()


def test_boot_pipe_truncates_oversized_traceback():
    """A bootstrap traceback larger than the pipe capacity must be truncated
    before the one-shot send, so the dying child never blocks in write() (which
    would wedge the node in 'creating' with no error ever surfaced)."""
    spec = dataclasses.replace(CATALOG["ConstantArray"], module="tests._huge_error", cls_name="HugeError")
    ref = spawn_node(spec)
    try:
        assert wait_until(lambda: ref.process is not None and not ref.process.is_alive(), timeout=5.0), (
            "child blocked writing the oversized traceback"
        )
        assert ref.boot_conn is not None and ref.boot_conn.poll(1.0)
        tb = ref.boot_conn.recv()
        assert "truncated" in tb
        assert "RuntimeError" in tb  # the exception type still survives the truncation
        assert len(tb) < 40000
    finally:
        ref.terminate()


def test_instance_error_tolerates_concurrent_container_mutation():
    """_instance_error runs on a NodeRef messaging thread and races structural
    RPCs mutating the container on executor threads; it must snapshot + guard,
    not iterate the live dict (which raises 'changed size'/KeyError)."""
    from types import SimpleNamespace

    from goofi.manager import Manager

    class _Nodes:
        def __iter__(self):
            return iter(["a", "b"])  # 'a' vanished between snapshot and lookup

        def __getitem__(self, k):
            if k == "a":
                raise KeyError(k)
            return SimpleNamespace(last_error="boom")

    fake = SimpleNamespace(nodes=_Nodes(), _within_subtree=lambda u, i: True)
    assert Manager._instance_error(fake, "inst") == "boom"


def test_nodeprocess_registry_get_is_atomic_under_concurrency(monkeypatch):
    """Concurrent get() for one group name must yield a single host — the
    unlocked check-then-create would fork a duplicate host per racing caller."""
    import threading
    import time as _time

    import goofi.node_helpers as nh

    nh._process_groups.clear()
    created = []

    class _StubHost:
        def __init__(self, name, iid, capture_logs=False):
            _time.sleep(0.02)  # simulate the real fork window where the race lives
            created.append(name)

        def kill(self):
            pass

    monkeypatch.setattr(nh, "NodeProcess", _StubHost)
    barrier = threading.Barrier(16)
    hosts = []

    def do():
        barrier.wait()
        hosts.append(nh.NodeProcessRegistry().get("grp", "iid"))

    threads = [threading.Thread(target=do) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1, f"get() created {len(created)} hosts under concurrency"
    assert len({id(h) for h in hosts}) == 1


def test_nodeprocess_spawn_serializes_concurrent_callers():
    """Two callers spawning into one host must not interleave their send/recv on
    the shared setup pipe (which would swap responses)."""
    import threading
    import time as _time

    from goofi.node_helpers import NodeProcess

    state = {"depth": 0, "violation": False}

    class _Conn:
        def send(self, payload):
            state["depth"] += 1
            if state["depth"] > 1:
                state["violation"] = True
            _time.sleep(0.01)

        def recv(self):
            _time.sleep(0.01)
            state["depth"] -= 1
            return ("ok", "id")

    host = NodeProcess.__new__(NodeProcess)
    host.name = "g"
    host.conn = _Conn()
    host._lock = threading.Lock()  # the fix initializes this in __init__

    threads = [threading.Thread(target=lambda i=i: host.spawn("m", "c", f"n{i}", None)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not state["violation"], "concurrent spawn transactions interleaved on the pipe"


def test_restart_consumer_does_not_double_register_producer():
    """Restarting a CONSUMER must not re-register the surviving producer's
    subscriber (its registration from the original wiring is still valid) —
    else the producer publishes forever for a subscriber count that never
    unwinds."""
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        prod = mgr.add_node("ConstantArray", "inputs", params={"common": {"autotrigger": True}})
        cons = mgr.add_node("Buffer", "signal")
        mgr.add_link(prod, cons, "out", "val")
        assert wait_until(
            lambda: (mgr.nodes[prod].serialized_state or {}).get("output_subscribers", {}).get("out", 0) == 1
        )

        mgr.nodes[cons].process.kill()
        assert wait_until(lambda: not mgr.nodes[cons].process.is_alive())
        mgr._supervise_once()
        assert wait_until(lambda: mgr.nodes[cons].stage == "ready")

        # After the rewire settles, the producer still has exactly ONE registered
        # subscriber for the one link — not two.
        time.sleep(0.6)
        assert (mgr.nodes[prod].serialized_state or {}).get("output_subscribers", {}).get("out", 0) == 1
    finally:
        mgr.terminate(notify_gui=False)


def test_pending_ctrl_coalesces_node_directory():
    """The pre-ready ctrl queue must coalesce NODE_DIRECTORY pushes (the node
    replaces its directory wholesale, so only the newest matters). A big patch
    load broadcasts the directory once per add; without coalescing a slow-setup
    node accumulates one per add and the flush hits the bounded (64) ctrl
    channel's wall — a stall or, if the node then crashes, a permanent wedge."""
    from goofi.message import Message, MessageType
    from goofi.node import ref_from_spec
    from goofi.transport import set_instance_id

    set_instance_id(f"t-coalesce-{id(object())}")
    spec = CATALOG["ConstantArray"]
    _, _, params = spec.configure()
    ref = ref_from_spec(spec, "coalesce-test", params, in_process=False, process=None)
    try:
        for i in range(100):  # 100 directory pushes while pre-ready
            ref._send(Message(MessageType.NODE_DIRECTORY, {"directory": {"n": str(i)}}))
        # an order-sensitive message must NOT be coalesced away
        ref._send(
            Message(
                MessageType.SUBSCRIBE_INPUT,
                {"slot_name_in": "val", "service_name": "svc", "in_process": False},
            )
        )
        dirs = [m for m in ref._pending_ctrl if m.type == MessageType.NODE_DIRECTORY]
        assert len(dirs) == 1, f"expected one coalesced directory, got {len(dirs)}"
        assert dirs[0].content["directory"] == {"n": "99"}  # newest wins
        assert len(ref._pending_ctrl) == 2  # 1 directory + the subscribe
    finally:
        ref.terminate()


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
