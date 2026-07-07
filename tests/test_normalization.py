"""TableNormalization's reset must reach its per-column child normalizers.

Fix #10 (serialize() emits False for trigger BoolParams) means the wrapper's
momentary `reset` can no longer cross to the children through
`self.params.serialize()` — it must be propagated explicitly.
"""
import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.signal.normalization import TableNormalization


def test_tablenormalization_reset_clears_child_buffers():
    node = TableNormalization.create_standalone()
    node.setup()

    def tick():
        table = Data(DataType.TABLE, {"a": Data(DataType.ARRAY, np.ones(10), {})}, {})
        node.process(table)

    tick()
    assert node.normalizers["a"].buffer.shape[-1] == 10
    tick()  # no reset -> the child buffer grows
    assert node.normalizers["a"].buffer.shape[-1] == 20

    node.params.normalization.reset.value = True  # click the reset trigger
    tick()
    assert node.normalizers["a"].buffer.shape[-1] == 10, "reset never reached the child normalizer"
