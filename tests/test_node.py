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
        assert ref.wait_for_state(timeout=3.0), "no STATE_UPDATE arrived from child process"
    finally:
        ref.terminate()
        ref.process.join(timeout=2.0)
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
