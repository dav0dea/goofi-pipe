import numpy as np

from goofi.audio.clock import SampleClock
from goofi.nodes.inputs.oscillator import Oscillator

from .audio_utils import FixedClock


def _render(clock_sizes, freq=440.0, sfreq=48000.0):
    node = Oscillator.create_standalone()
    node.setup()
    node.clock = FixedClock(clock_sizes)
    node.params.oscillator.sampling_frequency.value = sfreq
    node.params.oscillator.frequency.value = freq
    node.params.oscillator.type.value = "sine"
    return np.concatenate([node.process(None)["out"][0] for _ in clock_sizes])


def test_two_blocks_equal_one_block_phase_continuous():
    two = _render([128, 128])
    one = _render([256])
    assert two.shape == one.shape == (256,)
    np.testing.assert_allclose(two, one, atol=1e-6)


def test_block_size_follows_the_clock():
    node = Oscillator.create_standalone()
    node.setup()
    node.clock = FixedClock([300, 0, 512])
    assert node.process(None)["out"][0].shape[0] == 300
    assert node.process(None)["out"][0].size == 0          # clock says 0 -> empty block
    assert node.process(None)["out"][0].shape[0] == 512


def test_setup_creates_a_started_sample_clock():
    node = Oscillator.create_standalone()
    node.setup()
    assert isinstance(node.clock, SampleClock)
    node.process(None)
    assert node.clock.emitted >= 0


def test_output_stamps_sfreq():
    node = Oscillator.create_standalone()
    node.setup()
    node.params.oscillator.sampling_frequency.value = 48000.0
    node.clock = FixedClock([64])
    _, meta = node.process(None)["out"]
    assert meta["sfreq"] == 48000.0
