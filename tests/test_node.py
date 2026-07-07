"""Tests for the rewritten `Node` class and `NodeRef`."""
import time
import uuid

import pytest
import yaml

from goofi.data import DataType
from goofi.message import Message, MessageType
from goofi.node import Node
from goofi.node_helpers import InputSlot, NodeRef, OutputSlot
from goofi.params import DEFAULT_PARAMS, NodeParams
from goofi.transport import set_instance_id

from .utils import DummyNode, FullDummyNode, list_param_types, make_custom_node


def _iid():
    set_instance_id(f"t-{uuid.uuid4().hex[:8]}")


def test_handle_ctrl_refresh_param_recomputes_options():
    """REFRESH_PARAM calls the param's refresh method, sets the fresh options
    (keeping the current value selectable), and marks dirty so the next
    STATE_UPDATE carries the new option list."""
    from goofi.params import StringParam

    cls = make_custom_node(params={"g": {"pick": StringParam("a", options=["a"], refresh="_reload")}})
    node = cls.create_standalone()
    node._reload = lambda: ["x", "y"]  # bind a refresh method on the instance
    node._dirty = False

    node._handle_ctrl(Message(MessageType.REFRESH_PARAM, {"group": "g", "param_name": "pick"}))

    p = node.params["g"]["pick"]
    # the current value 'a' is not in the fresh list -> kept selectable at the front
    assert p.options == ["a", "x", "y"]
    assert node._dirty is True


def test_handle_ctrl_refresh_param_current_value_in_list():
    from goofi.params import StringParam

    cls = make_custom_node(params={"g": {"pick": StringParam("x", options=["x"], refresh="_reload")}})
    node = cls.create_standalone()
    node._reload = lambda: ["x", "y"]

    node._handle_ctrl(Message(MessageType.REFRESH_PARAM, {"group": "g", "param_name": "pick"}))

    # current value already present -> no duplicate prepend
    assert node.params["g"]["pick"].options == ["x", "y"]


def test_handle_ctrl_coerces_fractional_int_param():
    """The node-side param write must coerce to the param's type: a fractional
    value pushed to an IntParam is truncated, never stored as a float (which
    would later serialize into an unloadable patch)."""
    from goofi.params import IntParam

    node = make_custom_node(params={"g": {"n": IntParam(4, 0, 10)}}).create_standalone()
    node._handle_ctrl(
        Message(MessageType.PARAMETER_UPDATE, {"group": "g", "param_name": "n", "param_value": 2.5})
    )
    assert node.params["g"]["n"].value == 2


def test_noderef_update_param_coerces_fractional_int():
    """The manager-side mirror (NodeRef.update_param) must coerce too, so a
    fractional value typed into an int field never poisons the saved patch."""
    from goofi.params import IntParam

    _iid()
    cls = make_custom_node(params={"g": {"n": IntParam(4, 0, 10)}})
    ref, _ = cls.create_local()
    try:
        ref.update_param("g", "n", 2.5)
        assert ref.params["g"]["n"].value == 2
    finally:
        ref.terminate()


def test_noderef_update_param_rejects_uncoercible_value():
    from goofi.params import IntParam

    _iid()
    cls = make_custom_node(params={"g": {"n": IntParam(4, 0, 10)}})
    ref, _ = cls.create_local()
    try:
        with pytest.raises(ValueError):
            ref.update_param("g", "n", "not a number")
        assert ref.params["g"]["n"].value == 4  # unchanged
    finally:
        ref.terminate()


def test_abstract_node():
    """Instantiating an abstract Node subclass must raise TypeError."""
    with pytest.raises(TypeError):
        Node("nope", {}, {}, NodeParams(DEFAULT_PARAMS), None)


def test_create_local_basic():
    _iid()
    ref, n = DummyNode.create_local()
    assert n.alive
    assert len(n.input_slots) == 0
    assert len(n.output_slots) == 0
    assert n.params == NodeParams(DEFAULT_PARAMS)
    assert n.messaging_thread.is_alive()
    assert n.processing_thread.is_alive()
    ref.terminate()


def test_processing_error_clears_after_recovery():
    """A transient process() failure must not stick. While process() keeps
    raising, the node reports an error; once a later tick succeeds, the node
    must report the error cleared (None) rather than leaving it latched."""
    _iid()
    state = {"fail": True}

    def cb(**kwargs):
        if state["fail"]:
            raise ValueError("boom")
        return {}

    Custom = make_custom_node(process_callback=cb)
    # autotrigger free-runs the processing loop so it ticks without inputs.
    ref, n = Custom.create_local(initial_params={"common": {"autotrigger": True}})
    try:
        # While failing, the error is reported and stays observable.
        deadline = time.time() + 2.0
        while ref.last_error is None and time.time() < deadline:
            time.sleep(0.01)
        assert ref.last_error is not None and "boom" in ref.last_error

        # Recover: subsequent successful ticks must clear the error.
        state["fail"] = False
        deadline = time.time() + 2.0
        while ref.last_error is not None and time.time() < deadline:
            time.sleep(0.01)
        assert ref.last_error is None, f"error stuck after recovery: {ref.last_error!r}"
    finally:
        ref.terminate()


def test_terminate_via_ref():
    _iid()
    ref, n = DummyNode.create_local()
    # Initial state push should arrive shortly.
    assert ref.wait_for_state(timeout=1.0)
    ref.terminate()
    # Allow the messaging loop to consume TERMINATE and call cleanup.
    for _ in range(50):
        if not n.alive:
            break
        time.sleep(0.02)
    assert not n.alive, "Node should be dead after ref.terminate()."


def test_multiproc_create():
    _iid()
    ref = DummyNode.create()
    try:
        assert ref.process.is_alive()
        # The first STATE_UPDATE proves the spawned process came up.
        # Generous timeout — spawn + iceoryx2 init can take a moment on
        # a busy CI / dev machine.
        assert ref.wait_for_state(timeout=10.0), "no STATE_UPDATE arrived from child process"
    finally:
        ref.terminate()
        ref.process.join(timeout=5.0)
    assert not ref.process.is_alive()


def test_full_node_input_slots():
    _iid()
    ref, n = FullDummyNode.create_local()
    for name, slot in ref.input_slots.items():
        assert isinstance(slot, DataType)
        assert name in n.input_slots
        assert isinstance(n.input_slots[name], InputSlot)
        assert slot == n.input_slots[name].dtype
    ref.terminate()


def test_full_node_output_slots():
    _iid()
    ref, n = FullDummyNode.create_local()
    for name, slot in ref.output_slots.items():
        assert isinstance(slot, DataType)
        assert name in n.output_slots
        assert isinstance(n.output_slots[name], OutputSlot)
        assert slot == n.output_slots[name].dtype
    ref.terminate()


def test_full_node_params():
    _iid()
    ref, n = FullDummyNode.create_local()
    assert isinstance(ref.params, NodeParams)
    assert "common" in ref.params
    assert "test" in ref.params
    for param_type in list_param_types():
        name = "param_" + param_type.__name__.lower()
        assert name in ref.params.test
        assert isinstance(getattr(ref.params.test, name), param_type)
    ref.terminate()


def test_register_subscriber_increments_count():
    _iid()
    cls = make_custom_node(output_slots={"out": DataType.ARRAY})
    ref, n = cls.create_local()
    assert n.output_slots["out"].subscriber_count == 0
    ref.register_subscriber("out")
    # control message takes a tick to process
    for _ in range(50):
        if n.output_slots["out"].subscriber_count == 1:
            break
        time.sleep(0.02)
    assert n.output_slots["out"].subscriber_count == 1
    ref.unregister_subscriber("out")
    for _ in range(50):
        if n.output_slots["out"].subscriber_count == 0:
            break
        time.sleep(0.02)
    assert n.output_slots["out"].subscriber_count == 0
    ref.terminate()


@pytest.mark.parametrize("value", [10.0, 100.0])
def test_change_parameter(value):
    _iid()
    ref, n = DummyNode.create_local()
    ref.wait_for_state(timeout=1.0)
    ref.update_param("common", "max_frequency", value)
    for _ in range(50):
        if n.params.common.max_frequency.value == value:
            break
        time.sleep(0.02)
    assert n.params.common.max_frequency.value == value
    ref.terminate()


@pytest.mark.parametrize("value", [10.0, 100.0])
def test_change_parameter_callback(value):
    _iid()
    results = []

    def callback(_self, v):
        results.append(v)

    Cls = type("CallbackDummyNode", (DummyNode,), {"common_max_frequency_changed": callback})
    ref, _ = Cls.create_local()
    ref.wait_for_state(timeout=1.0)
    ref.update_param("common", "max_frequency", value)
    for _ in range(50):
        if results:
            break
        time.sleep(0.02)
    assert results == [value]
    ref.terminate()


def test_state_push_serialization():
    """Node pushes STATE_UPDATE on dirty; NodeRef caches it."""
    _iid()
    ref, n = FullDummyNode.create_local()
    assert ref.wait_for_state(timeout=2.0), "no STATE_UPDATE received from node"
    state = ref.serialized_state
    assert state is not None
    assert state["_type"] == "FullDummyNode"
    assert state["category"] == "tests"
    assert isinstance(state["params"], dict)
    assert "output_subscribers" in state
    ref.terminate()


def test_state_yaml_dumpable():
    """STATE_UPDATE content must be YAML-friendly so the manager can save patches."""
    _iid()
    ref, _ = FullDummyNode.create_local()
    assert ref.wait_for_state(timeout=2.0)
    state = dict(ref.serialized_state)
    state.pop("output_subscribers", None)
    out = yaml.dump(state, sort_keys=False)
    back = yaml.load(out, Loader=yaml.FullLoader)
    assert back == state
    ref.terminate()


def test_set_expression_round_trip():
    """SET_EXPRESSION ctrl roundtrips: source + enabled + flags + value
    all land on the node and echo back via STATE_UPDATE."""
    _iid()
    ref, n = FullDummyNode.create_local()
    ref.wait_for_state(timeout=2.0)
    # Bind an enabled expression with both cadence flags on.
    ref.set_expression(
        "test",
        "param_floatparam",
        "1.5 * 2",
        enabled=True,
        triggers_process=True,
        autoeval=True,
    )
    for _ in range(80):
        if n.params.test.param_floatparam._value == 3.0:
            break
        time.sleep(0.02)
    assert n.params.test.param_floatparam._value == 3.0
    assert n.params.test.param_floatparam.expression == "1.5 * 2"
    assert n.params.test.param_floatparam.expression_enabled is True
    assert n.params.test.param_floatparam.expression_triggers_process is True
    assert n.params.test.param_floatparam.expression_autoeval is True
    ref.terminate()


def test_expression_disable_preserves_source():
    """Disabling an expression (enabled=False, same source) tears down
    the engine but keeps the source on the Param so a later re-enable
    restores it without losing what the user wrote."""
    _iid()
    ref, n = FullDummyNode.create_local()
    ref.wait_for_state(timeout=2.0)
    src = "42"
    # Enable.
    ref.set_expression("test", "param_floatparam", src, enabled=True)
    for _ in range(80):
        if n.params.test.param_floatparam.expression_enabled:
            break
        time.sleep(0.02)
    assert n.params.test.param_floatparam.expression == src
    assert n.params.test.param_floatparam.expression_enabled is True
    # Disable but pass the same source: stash, no engine running.
    ref.set_expression("test", "param_floatparam", src, enabled=False)
    for _ in range(80):
        if not n.params.test.param_floatparam.expression_enabled:
            break
        time.sleep(0.02)
    assert n.params.test.param_floatparam.expression == src, (
        "source must survive a disable so fx toggle is reversible"
    )
    assert n.params.test.param_floatparam.expression_enabled is False
    # Re-enable with the stashed source.
    ref.set_expression("test", "param_floatparam", src, enabled=True)
    for _ in range(80):
        if n.params.test.param_floatparam.expression_enabled:
            break
        time.sleep(0.02)
    assert n.params.test.param_floatparam.expression_enabled is True
    # Truly clear by sending None.
    ref.set_expression("test", "param_floatparam", None, enabled=False)
    for _ in range(80):
        if n.params.test.param_floatparam.expression is None:
            break
        time.sleep(0.02)
    assert n.params.test.param_floatparam.expression is None
    ref.terminate()


def test_node_releases_ipc_endpoints_on_shutdown():
    """A terminated node must close its iceoryx2 endpoints, not leak them.

    This matters most for process-group hosting: the host process outlives
    any single node, so a node that never releases its publisher slot would
    block the name (and any reuse of it) from ever working again. We drive a
    node with iceoryx2 ctrl/status endpoints (in_process_with_manager=False)
    directly in this process and assert the publisher/subscriber handles are
    dropped once the messaging loop exits."""
    from goofi.node import NodeEnv

    _iid()
    in_slots, out_slots, params = DummyNode._configure()
    node = DummyNode(
        f"dummy-cleanup-{uuid.uuid4().hex[:8]}",
        in_slots,
        out_slots,
        params,
        NodeEnv.LOCAL,
        in_process_with_manager=False,
    )
    # Let the messaging/processing threads and setup come up.
    deadline = time.time() + 2.0
    while not node._setup_done.is_set() and time.time() < deadline:
        time.sleep(0.01)
    status_pub = node.status_pub
    ctrl_sub = node.ctrl_sub
    assert status_pub._pub is not None
    assert ctrl_sub._sub is not None

    # Terminate: the messaging loop notices and runs the node's teardown.
    node._alive = False
    node.messaging_thread.join(timeout=3.0)
    assert not node.messaging_thread.is_alive()
    assert status_pub._pub is None, "status publisher not released on shutdown"
    assert ctrl_sub._sub is None, "ctrl subscriber not released on shutdown"


def test_node_reresolves_expression_when_directory_updates():
    """A node must re-resolve cross-node expression references when the
    name->id directory arrives. This is the patch-load ordering: a node
    warm-evaluates `nd('producer')` during setup() before the producer's id
    is known, wiring to a dangling service; the later NODE_DIRECTORY push must
    re-key the subscription onto the producer's real transport id."""
    _iid()
    ref, n = FullDummyNode.create_local()
    ref.wait_for_state(timeout=2.0)
    ref.set_expression("test", "param_floatparam", "nd('producer').out\n0", enabled=True)
    key = ("test", "param_floatparam")

    def _wait(pred, timeout=2.0):
        deadline = time.time() + timeout
        while not pred() and time.time() < deadline:
            time.sleep(0.02)

    # Warm eval subscribes under the literal name (directory still empty).
    _wait(lambda: key in n._expressions and ("producer", "out") in n._expressions[key]._subscribed)
    assert ("producer", "out") in n._expressions[key]._subscribed

    # The directory now carries the producer's real id; the node re-resolves.
    ref.send_directory({"producer": "producer-deadbeef"})
    _wait(lambda: ("producer-deadbeef", "out") in n._expressions[key]._subscribed)
    assert ("producer-deadbeef", "out") in n._expressions[key]._subscribed
    assert ("producer", "out") not in n._expressions[key]._subscribed
    ref.terminate()


def test_match_fired_expression_tolerates_concurrent_mutation():
    """Report H4: the processing thread matches a fired listener against
    `self._expressions` while the messaging thread can SET/clear an expression
    (mutating the dict). Iterating a live dict would raise
    `RuntimeError: dictionary changed size during iteration`; the match must
    snapshot first."""

    import threading

    class _Stub:
        pass

    stub = _Stub()
    stub._expression_lock = threading.RLock()
    applied: list = []
    stub._apply_expression = lambda key, engine: applied.append(key)

    class _BenignEngine:
        def owns_listener(self, listener):
            return False

    class _MutatingEngine:
        """Simulates a concurrent SET_EXPRESSION: pops a sibling key while the
        processing thread is mid-iteration over `_expressions`."""

        def owns_listener(self, listener):
            stub._expressions.pop(("g", "c"), None)
            return False

    stub._expressions = {
        ("g", "a"): _MutatingEngine(),
        ("g", "b"): _BenignEngine(),
        ("g", "c"): _BenignEngine(),
    }

    # Bind the real method to the stub; must not raise despite the mid-loop pop.
    matched = Node._match_fired_expression(stub, object())
    assert matched is False
    assert applied == []
