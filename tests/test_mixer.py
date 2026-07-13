import numpy as np

from goofi.data import Data, DataType
from goofi.nodes.signal.mixer import Mixer


def _blk(vals, index, sfreq=48000):
    return Data(DataType.ARRAY, np.asarray(vals, dtype=np.float32), {"sfreq": sfreq, "index": index})


def _out(res):
    return None if res is None else res["out"][0]


def test_inputs_are_queue_slots():
    node = Mixer.create_standalone()
    assert all(s.queue for s in node.input_slots.values())


def test_mixer_is_input_triggered_not_free_running():
    # Its queue inputs drive it: each delivered frame triggers process(). With the
    # unbounded default max_frequency (0) it runs at the producers' rate without
    # free-running. autotrigger would only busy-run the loop when NO input is
    # connected (nothing to mix anyway).
    node = Mixer.create_standalone()
    assert node.params.common.autotrigger.value is False


def test_averages_aligned_blocks():
    node = Mixer.create_standalone()
    node.setup()
    res = node.process(in0=_blk([1.0, 3.0, 5.0], 0), in1=_blk([3.0, 5.0, 7.0], 0))
    np.testing.assert_allclose(_out(res), [2.0, 4.0, 6.0])   # (a+b)/2 over 2 active inputs


def test_sum_mode_adds_instead_of_averaging():
    node = Mixer.create_standalone()
    node.setup()
    node.params.mixer.normalize.value = False
    res = node.process(in0=_blk([1.0, 2.0], 0), in1=_blk([10.0, 20.0], 0))
    np.testing.assert_allclose(_out(res), [11.0, 22.0])


def test_single_active_input_passes_through():
    node = Mixer.create_standalone()
    node.setup()
    res = node.process(in0=_blk([4.0, 6.0], 0))
    np.testing.assert_allclose(_out(res), [4.0, 6.0])        # avg over 1 active = itself


def test_stale_reread_is_deduped_by_index():
    # A repeated frame (same index) must not be written twice.
    node = Mixer.create_standalone()
    node.setup()
    res1 = node.process(in0=_blk([2.0, 2.0], 5))
    np.testing.assert_allclose(_out(res1), [2.0, 2.0])       # first delivery emits
    res2 = node.process(in0=_blk([2.0, 2.0], 5))             # same index -> nothing new
    assert _out(res2) is None                                # ring drained, no re-write


def test_ragged_lengths_align_on_min_fill():
    # Independent generators emit different per-tick counts; the rings absorb the
    # difference and only the common (min) frame count is emitted, the rest waits.
    node = Mixer.create_standalone()
    node.setup()
    res = node.process(in0=_blk([1.0, 1.0, 1.0, 1.0], 0), in1=_blk([3.0, 3.0], 0))
    np.testing.assert_allclose(_out(res), [2.0, 2.0])        # min fill = 2 frames
    # in0's two surplus frames stay buffered until in1 supplies more (stay aligned).
    res2 = node.process(in1=_blk([3.0, 3.0], 1))
    np.testing.assert_allclose(_out(res2), [2.0, 2.0])
