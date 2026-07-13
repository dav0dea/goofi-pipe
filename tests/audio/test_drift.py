import numpy as np

from goofi.audio.drift import DriftCorrector


def _sine(n, ch=1):
    x = np.sin(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)).astype(np.float32)
    return x[:, None].repeat(ch, axis=1)


def test_within_deadband_is_passthrough_identity():
    d = DriftCorrector(1000, deadband=64)
    block = _sine(480)
    out = d.correct(block, fill=1000)          # exactly target
    assert out is block
    out2 = d.correct(block, fill=1050)         # inside +deadband
    assert out2 is block


def test_empty_block_is_passthrough_out_of_band():
    # A zero-length block has nothing to stuff/drop and _zero_crossing (argmin over
    # an empty array) is undefined. Generators emit empty frames by design when the
    # SampleClock has no whole sample due, and they reach the sink out-of-band.
    d = DriftCorrector(240, deadband=64)
    empty = np.zeros((0, 2), dtype=np.float32)
    assert d.correct(empty, fill=10).shape == (0, 2)     # underfull → would have stuffed
    assert d.correct(empty, fill=400).shape == (0, 2)    # overfull → would have dropped


def test_overfull_drops_one_frame_at_zero_crossing():
    d = DriftCorrector(1000, deadband=64)
    block = np.full((100, 1), 0.5, dtype=np.float32)
    block[50, 0] = 0.0                          # the single zero-crossing frame
    out = d.correct(block, fill=2000)
    assert out.shape[0] == 99                   # len - 1
    assert not np.any(out == 0.0)               # the near-zero frame was the one dropped
    assert np.all(out == 0.5)


def test_underfull_duplicates_one_frame_at_zero_crossing():
    d = DriftCorrector(1000, deadband=64)
    block = np.full((100, 1), 0.5, dtype=np.float32)
    block[50, 0] = 0.0
    out = d.correct(block, fill=0)
    assert out.shape[0] == 101                  # len + 1
    assert np.count_nonzero(out == 0.0) == 2    # the zero frame duplicated


def test_energy_preserved_on_drop_and_dup():
    d = DriftCorrector(1000, deadband=64)
    block = _sine(480)
    e0 = float(np.sum(block ** 2))
    dropped = d.correct(block, fill=5000)
    dupped = d.correct(block, fill=0)
    assert dropped.shape[0] == 479
    assert dupped.shape[0] == 481
    # edits land on the near-zero sample, so total energy barely moves
    assert abs(float(np.sum(dropped ** 2)) - e0) < 1e-3
    assert abs(float(np.sum(dupped ** 2)) - e0) < 1e-3


def test_multichannel_len_pm_one():
    d = DriftCorrector(1000, deadband=64)
    block = _sine(256, ch=2)
    assert d.correct(block, fill=9000).shape == (255, 2)
    assert d.correct(block, fill=0).shape == (257, 2)
