import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.signal.gain import Gain


def test_gain_scales_and_propagates_sfreq():
    node = Gain.create_standalone()
    node.setup()
    node.params.gain.gain.value = 2.0
    sig = Data(DataType.ARRAY, np.array([1.0, -2.0, 3.0]), {"sfreq": 48000})
    out, meta = node.process(sig)["out"]
    np.testing.assert_allclose(out, [2.0, -4.0, 6.0])
    assert meta["sfreq"] == 48000


def test_gain_returns_none_without_input():
    node = Gain.create_standalone()
    node.setup()
    assert node.process(None) is None


def test_gain_input_is_a_queue_slot():
    node = Gain.create_standalone()
    assert node.input_slots["signal"].queue is True
