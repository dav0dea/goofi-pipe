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


def test_noderef_carries_spec_not_class():
    """The manager-side proxy is spec-shaped: it must never hold the node CLASS
    (holding the class means the manager imported the implementation)."""
    from goofi.nodes.inputs.constantarray import ConstantArray

    ref, node = ConstantArray.create_local(initial_params={"common": {"autotrigger": False}})
    try:
        assert not hasattr(ref, "node_class")
        assert ref.spec.type == "ConstantArray"
        assert ref.spec.cls_name == "ConstantArray"
        assert ref.spec.module == "goofi.nodes.inputs.constantarray"
        assert ref.category == ref.spec.category == "inputs"
        assert ref.__doc__ == ref.spec.doc
    finally:
        ref.terminate()
