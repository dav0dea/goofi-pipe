import pytest

from goofi.message import Message, MessageType


EXAMPLE_CONTENT = {
    MessageType.SUBSCRIBE_INPUT: {
        "slot_name_in": "in",
        "service_name": "svc",
        "in_process": False,
    },
    MessageType.UNSUBSCRIBE_INPUT: {"slot_name_in": "in"},
    MessageType.REGISTER_SUBSCRIBER: {"slot_name_out": "out"},
    MessageType.UNREGISTER_SUBSCRIBER: {"slot_name_out": "out"},
    MessageType.PARAMETER_UPDATE: {"group": "common", "param_name": "x", "param_value": "y"},
    # expression field is optional (None == clear binding); the test
    # below only verifies missing-required-field raises.
    MessageType.SET_EXPRESSION: {"group": "common", "param_name": "x"},
    MessageType.CLEAR_DATA: {"slot_name": "in"},
    MessageType.TERMINATE: {},
    MessageType.STATE_UPDATE: {
        "_type": "Foo",
        "category": "test",
        "params": {},
        "output_subscribers": {},
    },
    MessageType.PROCESSING_ERROR: {"error": "boom"},
    MessageType.SHUTDOWN: {},
}


@pytest.mark.parametrize("type", MessageType.__members__.values())
def test_create_message(type):
    if type not in EXAMPLE_CONTENT:
        raise NotImplementedError(f"Missing test for {type}.")
    Message(type, EXAMPLE_CONTENT[type])


@pytest.mark.parametrize("type", MessageType.__members__.values())
def test_message_content(type):
    if type not in EXAMPLE_CONTENT:
        raise NotImplementedError(f"Missing test for {type}.")
    for key in EXAMPLE_CONTENT[type]:
        content = EXAMPLE_CONTENT[type].copy()
        del content[key]
        # Messages with no required fields (e.g. TERMINATE) tolerate empty
        # content; skip the required-field check for those.
        if not content and type in (MessageType.TERMINATE, MessageType.SHUTDOWN):
            continue
        with pytest.raises(ValueError):
            Message(type, content)


@pytest.mark.parametrize("type", MessageType.__members__.values())
def test_message_errors(type):
    if type not in EXAMPLE_CONTENT:
        raise NotImplementedError(f"Missing test for {type}.")
    with pytest.raises(ValueError):
        Message(None, EXAMPLE_CONTENT[type])
    with pytest.raises(ValueError):
        Message(type, None)
