import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.outputs.audioout import AudioOut

from .audio_utils import fake_stream_factory


def _make_node():
    node = AudioOut.create_standalone()
    node.stream_factory = fake_stream_factory
    node.params.audio.sampling_rate.value = "48000"
    node.params.audio.buffer_ms.value = 5      # target_fill = 240, deadband 64
    node.params.audio.blocksize.value = 128
    node.setup()
    return node


def _audio(n, value, index=None):
    meta = {"sfreq": 48000}
    if index is not None:
        meta["index"] = index
    return Data(DataType.ARRAY, np.full(n, value, dtype=np.float32), meta)


def test_contiguous_index_never_crossfades():
    node = _make_node()
    node.process(_audio(128, 0.5, index=0), None)
    node.process(_audio(128, 0.5, index=1), None)
    node.process(_audio(128, 0.5, index=2), None)
    assert node.discontinuities == 0


def test_index_jump_triggers_one_crossfade():
    node = _make_node()
    node.process(_audio(128, 0.5, index=0), None)
    node.process(_audio(128, 0.5, index=1), None)
    node.process(_audio(128, -0.5, index=7), None)   # gap 1 -> 7
    assert node.discontinuities == 1


def test_missing_index_never_crossfades():
    node = _make_node()
    node.process(_audio(128, 0.5), None)             # no index key
    node.process(_audio(128, -0.5), None)
    assert node.discontinuities == 0


def test_drift_not_applied_before_playback_starts():
    # Drift correction is a steady-state mechanism: during priming the ring is
    # intentionally below target_fill, so applying it there would stuff a
    # duplicate frame into every pre-playback write (audible in the first audio
    # the user hears). Before `started`, blocks must pass through untouched.
    node = _make_node()                              # target_fill 240, deadband 64
    node.process(_audio(64, 0.0), None)              # fill 64 << target-deadband (176)
    assert node.started is False
    assert node.ring.fill == 64                      # exactly one block, no stuffed frame


def test_drift_drops_a_frame_when_ring_is_overfull():
    node = _make_node()
    for _ in range(3):
        node.process(_audio(128, 0.0), None)         # push fill past target+deadband, no pumping
    fill_before = node.ring.fill
    assert fill_before > node.target_fill + 64
    wave = np.sin(np.linspace(0.0, 4 * np.pi, 100)).astype(np.float32)
    node.process(Data(DataType.ARRAY, wave, {"sfreq": 48000}), None)
    assert node.ring.fill == fill_before + 99        # one frame dropped at a zero crossing
