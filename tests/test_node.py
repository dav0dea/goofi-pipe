import time
from multiprocessing import Manager as MPManager

import pytest
import yaml

from goofi.connection import Connection
from goofi.data import DataType
from goofi.message import Message, MessageType
from goofi.node import Node
from goofi.node_helpers import InputSlot, NodeRef, OutputSlot
from goofi.params import DEFAULT_PARAMS, NodeParams

from .utils import (
    DummyNode,
    FullDummyNode,
    ProcessingErrorNode,
    list_param_types,
    make_custom_node,
)

try:
    mp_manager = MPManager()
    Connection.set_backend("mp", mp_manager)
except AssertionError:
    # connection backend is already set
    pass


def test_abstract_node():
    # instantiating an abstract node should raise a TypeError
    with pytest.raises(TypeError):
        Node(Connection.create()[0])


def test_create_node():
    ref, n = DummyNode.create_local()

    # some basic checks
    assert n.alive, "Node is not alive."
    assert len(n.input_slots) == 0, "Node input slots are not empty."
    assert len(n.output_slots) == 0, "Node output slots are not empty."
    assert n.params == NodeParams(DEFAULT_PARAMS), "Node params are not default."
    assert n.messaging_thread.is_alive(), "Node messaging thread is not alive."
    assert n.processing_thread.is_alive(), "Node processing thread is not alive."

    # clean up dangling threads
    ref.terminate()


@pytest.mark.skip("This test is inconsistent")
def test_dead_pipe():
    ref, n = DummyNode.create_local()

    # close the connection
    ref.connection.close()
    time.sleep(0.05)

    # the node should stop its messaging thread and exit
    assert not n.messaging_thread.is_alive(), "Node messaging thread is alive."


def test_ping_pong():
    messages = []

    def callback(_: NodeRef, msg: Message):
        messages.append(msg)

    ref, _ = DummyNode.create_local()
    ref.set_message_handler(MessageType.PONG, callback)

    ref.connection.send(Message(MessageType.PING, {}))
    time.sleep(0.01)
    assert len(messages) == 1, "Node should have received one message."
    assert messages[0].type == MessageType.PONG, "Node should have received a pong message."

    # clean up dangling threads
    ref.terminate()


def test_terminate_message():
    ref, n = DummyNode.create_local()
    ref.connection.send(Message(MessageType.TERMINATE, {}))
    time.sleep(0.01)
    assert not n.alive, "Node should be dead after receiving terminate message."


def test_terminate_func():
    ref, n = DummyNode.create_local()
    ref.terminate()
    time.sleep(0.01)
    assert not n.alive, "Node should be dead after receiving terminate message."


def test_multiproc():
    ref = DummyNode.create()

    time.sleep(0.1)  # TODO: 0.1 is quite long but required on Windows, Linux works with 0.01
    # make sure the node is not dead
    assert ref.process.is_alive(), "Process should be alive."

    # send a terminate message
    ref.connection.send(Message(MessageType.TERMINATE, {}))
    time.sleep(0.1)  # TODO: 0.1 is quite long but required on Windows, Linux works with 0.01
    # make sure the node is dead
    assert not ref.process.is_alive(), "Process should be dead."


def test_processing_error(timeout: float = 1.5):
    messages = []

    def error_callback(ref: NodeRef, msg: Message):
        messages.append(msg)

    ref, n = ProcessingErrorNode.create_local()

    ref.set_message_handler(MessageType.PROCESSING_ERROR, error_callback)

    # manually trigger processing once
    n.process_flag.set()

    start = time.time()
    while len(messages) == 0:
        if time.time() - start > timeout:
            raise TimeoutError("Timeout while waiting for error message.")
        time.sleep(0.01)

    # make sure we received the error message
    assert len(messages) == 1, "Node should have received one message."

    # the processing thread should be alive as the node is expected to handle the error
    assert n.processing_thread.is_alive(), "Processing thread should still be alive."

    # the node and messaging thread should be alive
    assert n.alive, "Node should be alive."
    assert n.messaging_thread.is_alive(), "Messaging thread should be alive."

    # clean up dangling threads
    ref.terminate()


def test_full_node_input_slots():
    ref, n = FullDummyNode.create_local()

    for name, slot in ref.input_slots.items():
        assert isinstance(slot, DataType), f"NodeRef slots should be DataTypes, got {type(slot)}"
        assert name in n.input_slots, f"NodeRef input slot {name} is missing from node input slots."
        assert isinstance(
            n.input_slots[name], InputSlot
        ), f"Node input slot {name} should be an InputSlot, got {type(n.input_slots[name])}"
        assert slot == n.input_slots[name].dtype, f"Mismatching slot types for {name}: {slot} != {n.input_slots[name].dtype}"

    ref.terminate()


def test_full_node_output_slots():
    ref, n = FullDummyNode.create_local()

    for name, slot in ref.output_slots.items():
        assert isinstance(slot, DataType), f"NodeRef slots should be DataTypes, got {type(slot)}"
        assert name in n.output_slots, f"NodeRef output slot {name} is missing from node output slots."
        assert isinstance(
            n.output_slots[name], OutputSlot
        ), f"Node output slot {name} should be an OutputSlot, got {type(n.output_slots[name])}"
        assert slot == n.output_slots[name].dtype, f"Mismatching slot types for {name}: {slot} != {n.output_slots[name].dtype}"

    ref.terminate()


def test_full_node_params():
    ref, n = FullDummyNode.create_local()

    assert isinstance(ref.params, NodeParams), f"NodeRef params should be a NodeParams, got {type(ref.params)}"
    assert ref.params == n.params, "NodeRef params should be equal to node params."

    assert "common" in ref.params, "Node params should contain common params."
    assert "test" in ref.params, "Node params should contain test params."

    for param_type in list_param_types():
        name = "param_" + param_type.__name__.lower()
        assert name in ref.params.test, f"Node params should contain param '{name}'. Got {ref.params.test}"
        assert isinstance(
            getattr(ref.params.test, name), param_type
        ), f"Node param '{name}' should be a {param_type}, got {type(ref.params.test[name])}"

    ref.terminate()


def test_pipes():
    results = []

    def callback(**kwargs):
        results.append(kwargs["in"])

    cls1 = make_custom_node(output_slots={"out": DataType.ARRAY}, params={"common": {"autotrigger": True}})
    cls2 = make_custom_node(input_slots={"in": DataType.ARRAY}, process_callback=callback)

    ref1, n1 = cls1.create_local()
    ref2, _ = cls2.create_local()

    # connect the nodes
    ref1.connection.send(
        Message(
            MessageType.ADD_OUTPUT_PIPE,
            {"slot_name_out": "out", "slot_name_in": "in", "node_connection": ref2.connection},
        )
    )

    time.sleep(0.1)

    assert (
        len(n1.output_slots["out"].connections) == 2
    ), f"Expected two output slot connections, got {len(n1.output_slots['out'].connections)}"
    assert ("in", ref2.connection, False) in n1.output_slots["out"].connections, "Output slot connections are incorrect."
    assert len(results) > 0, "Processing callback should have been called once."

    # disconnect the nodes
    ref1.connection.send(
        Message(
            MessageType.REMOVE_OUTPUT_PIPE,
            {"slot_name_out": "out", "slot_name_in": "in", "node_connection": ref2.connection},
        )
    )

    time.sleep(0.01)

    assert len(n1.output_slots["out"].connections) == 1, "Output slot connections are incorrect."

    ref1.terminate()
    ref2.terminate()


@pytest.mark.parametrize("value", [10.0, 100.0])
def test_change_parameter(value):
    ref, n = DummyNode.create_local()

    # change a parameter
    ref.update_param("common", "max_frequency", value)

    time.sleep(0.01)

    # check that the parameter was changed
    assert n.params.common.max_frequency.value == value, "Parameter was not changed."

    # clean up dangling threads
    ref.terminate()


@pytest.mark.parametrize("value", [10.0, 100.0])
def test_change_parameter_callback(value):
    results = []

    def callback(_, value):
        results.append(value)

    # create a dummy node type with a callback for the common.max_frequency parameter
    CallbackDummyNode = type("CallbackDummyNode", (DummyNode,), {"common_max_frequency_changed": callback})
    ref, _ = CallbackDummyNode.create_local()

    # change a parameter
    ref.update_param("common", "max_frequency", value)

    time.sleep(0.01)

    # check that the parameter callback was called with the correct value
    assert len(results) == 1, "Parameter callback was not called."
    assert results[0] == value, "Parameter callback was called with the wrong value."

    # clean up dangling threads
    ref.terminate()


def test_serialize():
    result = None

    def callback(_, msg):
        nonlocal result
        result = msg.content

    ref, n = FullDummyNode.create_local()
    ref.set_message_handler(MessageType.SERIALIZE_RESPONSE, callback)

    # serialize the node
    ref.connection.send(Message(MessageType.SERIALIZE_REQUEST, {}))
    time.sleep(0.05)

    # "ground truth" serialized node
    node_type = "FullDummyNode"
    category = "tests"
    out_conns = {name: slot.connections for name, slot in n.output_slots.items()}
    params = n.params.serialize()

    # compare to the serialized node
    assert result["_type"] == node_type, "Node type is not serialized correctly."
    assert result["category"] == category, "Node category is not serialized correctly."
    assert result["out_conns"] == out_conns, "Output slots are not serialized correctly."
    assert result["params"] == params, "Params are not serialized correctly."

    # clean up dangling threads
    ref.terminate()


def test_serialize_yaml():
    result = None

    def callback(_, msg):
        nonlocal result
        result = msg.content

    ref, _ = FullDummyNode.create_local()
    ref.set_message_handler(MessageType.SERIALIZE_RESPONSE, callback)

    # serialize the node
    ref.connection.send(Message(MessageType.SERIALIZE_REQUEST, {}))
    time.sleep(0.01)

    # make sure the serialized parameters can be converted to YAML (except out_conns, which are removed in the manager)
    result.pop("out_conns")
    serialized = yaml.dump(result, sort_keys=False)
    reconstructed = yaml.load(serialized, Loader=yaml.FullLoader)

    assert reconstructed == result, "Serialized parameters are not valid YAML."

    # clean up dangling threads
    ref.terminate()
