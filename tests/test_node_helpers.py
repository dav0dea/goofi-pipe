import pytest

from goofi.data import DataType
from goofi.node_helpers import InputSlot, OutputSlot


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_input_slot(dtype):
    slot = InputSlot(dtype)
    assert slot.trigger_process is True
    assert slot.data is None
    # No live transport endpoints until SUBSCRIBE_INPUT is handled.
    assert slot.subscriber is None
    assert slot.listener is None


@pytest.mark.parametrize("dtype", DataType.__members__.values())
def test_output_slot(dtype):
    slot = OutputSlot(dtype)
    assert slot.subscriber_count == 0
    assert slot.publishers == []
    assert slot.notifiers == []
    assert slot.has_ipc is False
    assert slot.has_thread is False
