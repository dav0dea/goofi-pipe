"""Unit tests for SerialStream's dropout guard.

The serial protocols timestamp every packet on arrival and feed those
timestamps to ``np.interp`` as the x-coordinates. ``np.interp`` happily
draws a straight line across any gap in those x-coordinates, so a dropped
serial packet would be silently bridged and the resampled signal corrupted.
``_has_dropouts`` is the pure decision that lets ``process`` refuse to
interpolate across such a gap. These tests exercise that real logic
directly -- no serial hardware and no node spawn required.
"""

import numpy as np

from goofi.nodes.inputs.serialstream import _has_dropouts


def test_continuous_buffer_is_not_a_dropout():
    # Evenly spaced timestamps at the expected cadence -> safe to interpolate.
    time_buf = np.arange(0.0, 1.0, 0.01)
    assert _has_dropouts(time_buf, expected_dt=0.01) is False


def test_large_gap_is_a_dropout():
    # A jump far larger than the cadence means packets were dropped; the
    # interp would splice a straight line across the missing samples.
    time_buf = [0.0, 0.01, 0.02, 0.03, 0.50, 0.51, 0.52]
    assert _has_dropouts(time_buf, expected_dt=0.01) is True


def test_non_monotonic_buffer_is_a_dropout():
    # Time going backwards (or repeating) is invalid as interp x-coords.
    time_buf = [0.0, 0.01, 0.005, 0.02, 0.03]
    assert _has_dropouts(time_buf, expected_dt=0.01) is True


def test_small_jitter_within_tolerance_is_not_a_dropout():
    # Real arrival times jitter a little around the nominal cadence; that
    # must not trip the guard and stall the normal continuous path.
    time_buf = [0.0, 0.011, 0.019, 0.031, 0.039, 0.052]
    assert _has_dropouts(time_buf, expected_dt=0.01, tol=2.0) is False
