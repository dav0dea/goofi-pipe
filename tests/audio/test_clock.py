import math

from goofi.audio.clock import SampleClock


def test_start_resets_emitted():
    c = SampleClock(48000.0)
    c.start(123.0)
    assert c.emitted == 0


def test_sum_of_n_equals_floor_elapsed_over_irregular_ticks():
    sr = 48000.0
    c = SampleClock(sr)
    t0 = 100.0
    c.start(t0)
    gaps = [0.01, 0.005, 0.02, 0.013, 0.007, 0.03, 0.001]
    now = t0
    total = 0
    for g in gaps:
        now += g
        n = c.advance(now)
        assert n >= 0
        total += n
    assert total == math.floor((now - t0) * sr)
    assert c.emitted == total


def test_max_block_clamp_and_paced_resume():
    sr = 48000.0
    c = SampleClock(sr, max_block=8192, resync_after=0.25)
    c.start(0.0)
    # A 0.2 s gap wants 9600 frames; must clamp to max_block and realign.
    n = c.advance(0.2)
    assert n == 8192
    assert c.emitted == 8192
    # After the realign the clock resumes paced (drops the backlog).
    n2 = c.advance(0.21)
    assert n2 == math.floor(0.01 * sr)  # 480


def test_zero_and_backward_ticks_return_zero():
    c = SampleClock(48000.0)
    c.start(5.0)
    assert c.advance(5.0) == 0
    assert c.advance(4.5) == 0
    assert c.emitted == 0


def test_resync_after_long_stall_below_max_block():
    # raw stays under max_block but the clock is far behind -> resync branch.
    sr = 1000.0
    c = SampleClock(sr, max_block=100000, resync_after=0.25)
    c.start(0.0)
    assert c.advance(0.01) == 10
    n = c.advance(5.0)
    assert 0 <= n <= 100000
    # timebase realigned -> next modest tick is paced again, not catching up 5 s.
    assert c.advance(5.01) == math.floor(0.01 * sr)  # 10
