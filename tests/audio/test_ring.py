import numpy as np

from goofi.audio.ring import AudioRing


def _ramp(start, n, ch=1):
    return (start + np.arange(n, dtype=np.float32))[:, None].repeat(ch, axis=1)


def test_props_and_initial_state():
    r = AudioRing(8, 2)
    assert r.capacity == 8
    assert r.channels == 2
    assert r.fill == 0
    assert r.overruns == 0
    assert r.underruns == 0


def test_write_then_read_roundtrip():
    r = AudioRing(8, 1)
    assert r.write(_ramp(0, 3)) == 3
    assert r.fill == 3
    out = np.zeros((3, 1), dtype=np.float32)
    assert r.read_into(out) == 3
    assert r.fill == 0
    assert np.array_equal(out, _ramp(0, 3))
    assert r.underruns == 0


def test_wraparound_preserves_fifo_order():
    r = AudioRing(4, 1)
    r.write(_ramp(0, 3))            # w=3, fill=3
    out = np.zeros((2, 1), dtype=np.float32)
    r.read_into(out)               # r=2, fill=1  (frame 2 remains)
    assert r.write(_ramp(3, 3)) == 3  # crosses the boundary, fill=4
    got = np.zeros((4, 1), dtype=np.float32)
    assert r.read_into(got) == 4
    assert np.array_equal(got.ravel(), np.array([2, 3, 4, 5], dtype=np.float32))


def test_reject_newest_on_overflow():
    r = AudioRing(4, 1)
    accepted = r.write(_ramp(0, 6))   # only 4 fit; frames 4,5 rejected
    assert accepted == 4
    assert r.overruns == 1
    assert r.fill == 4
    got = np.zeros((4, 1), dtype=np.float32)
    r.read_into(got)
    assert np.array_equal(got.ravel(), np.array([0, 1, 2, 3], dtype=np.float32))


def test_partial_read_underrun_leaves_tail_untouched():
    r = AudioRing(8, 1)
    r.write(_ramp(0, 2))
    out = np.full((5, 1), -9.0, dtype=np.float32)
    assert r.read_into(out) == 2
    assert r.underruns == 1
    assert np.array_equal(out[:2].ravel(), np.array([0, 1], dtype=np.float32))
    assert np.all(out[2:] == -9.0)


def test_multichannel_and_spsc_fill_math():
    r = AudioRing(16, 2)
    total_w = 0
    total_r = 0
    for i in range(20):
        total_w += r.write(_ramp(i * 3, 3, ch=2))
        out = np.zeros((2, 2), dtype=np.float32)
        total_r += r.read_into(out)
    assert r.fill == total_w - total_r
    assert 0 <= r.fill <= r.capacity
