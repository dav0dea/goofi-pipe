import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.outputs.audioout import AudioOut

from .audio_utils import fake_stream_factory


def _make_node():
    node = AudioOut.create_standalone()
    node.stream_factory = fake_stream_factory
    node.params.audio.sampling_rate.value = "48000"
    node.params.audio.buffer_ms.value = 5      # target_fill = 240 frames at 48k
    node.params.audio.blocksize.value = 128
    node.setup()
    return node


def _audio(n, value):
    return Data(DataType.ARRAY, np.full(n, value, dtype=np.float32), {"sfreq": 48000})


def test_data_input_is_a_queue_slot():
    node = AudioOut.create_standalone()
    assert node.input_slots["data"].queue is True


def test_ring_capacity_holds_at_least_one_callback_block():
    # buffer_ms and blocksize are independent params. A low-latency / large-block
    # combo (target_fill*4 < blocksize) would leave the ring unable to hold even
    # one callback block -> read_into short-reads every callback (perpetual
    # underrun, mostly-zero output). Capacity must floor at blocksize + jitter.
    node = AudioOut.create_standalone()
    node.stream_factory = fake_stream_factory
    node.params.audio.sampling_rate.value = "16000"
    node.params.audio.buffer_ms.value = 5       # target_fill = 80 -> *4 = 320
    node.params.audio.blocksize.value = 2048     # >> 320
    node.setup()
    assert node.ring.capacity >= node.params.audio.blocksize.value


def test_primes_before_starting_the_stream():
    node = _make_node()
    node.process(_audio(128, 0.1), None)        # fill 128 < target 240
    assert node.started is False
    assert node.stream.started is False
    node.process(_audio(128, 0.1), None)        # fill 256 >= target -> prime + start
    assert node.started is True
    assert node.stream.started is True


def test_callback_outputs_ring_audio():
    node = _make_node()
    node.process(_audio(128, 0.25), None)
    node.process(_audio(128, 0.25), None)       # primed + started
    out = node.stream.pump(128)
    assert out.shape == (128, 2)
    assert np.allclose(out, 0.25)


def test_underrun_zero_fills_and_counts_without_touching_the_stream():
    node = _make_node()
    node.process(_audio(128, 0.25), None)
    node.process(_audio(128, 0.25), None)       # fill ~256, started
    before = node.ring.underruns
    node.stream.pump(128)                        # fill -> 128
    node.stream.pump(128)                        # fill -> 0
    node.stream.pump(128)                        # nothing left -> underrun + zeros
    assert node.ring.underruns > before
    assert np.allclose(node.stream.captured[-1], 0.0)
    assert node.stream.closed is False
