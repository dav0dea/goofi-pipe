"""The processing loop's max_frequency throttle, and its queue exemption.

A `queue` input is lossless-and-prompt by contract, so a tick that drained a
queue burst must NOT be rate-limited — otherwise the throttle re-introduces the
very burst gaps queue mode exists to remove (an audio processor scaling a 48 kHz
stream has to run at the producer's rate, not at the 30 Hz default cap)."""
import numpy as np

from goofi.data import Data, DataType
from tests.utils import make_custom_node


def _node():
    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY})
    return Cls.create_standalone()


def test_queue_drained_tick_is_exempt_from_throttle():
    n = _node()
    n.params.common.max_frequency.value = 30.0  # a real cap...
    frames = [Data(DataType.ARRAY, np.array([0.0]), {})]
    # ...but this tick drained a queue burst -> no throttle, run at input rate.
    assert n._rate_limit_sleep({"data": frames}, last_update=0.0, now=0.0) is None


def test_cap_off_is_unthrottled():
    n = _node()
    n.params.common.max_frequency.value = 0.0
    assert n._rate_limit_sleep({}, last_update=0.0, now=0.0) is None


def test_normal_tick_sleeps_the_remaining_period():
    n = _node()
    n.params.common.max_frequency.value = 30.0  # updates-per-second -> 1/30 s period
    # 10 ms into a 33.3 ms period -> ~23.3 ms left to sleep.
    sleep = n._rate_limit_sleep({}, last_update=0.0, now=0.010)
    assert abs(sleep - (1.0 / 30.0 - 0.010)) < 1e-9


def test_behind_schedule_returns_non_positive_no_sleep():
    n = _node()
    n.params.common.max_frequency.value = 30.0
    # Already 100 ms past a 33.3 ms period -> negative -> caller won't sleep.
    assert n._rate_limit_sleep({}, last_update=0.0, now=0.100) <= 0.0


def test_seconds_per_update_mode_uses_period_directly():
    n = _node()
    n.params.common.max_frequency.value = 2.0
    n.params.common.frequency_mode.value = "seconds-per-update"
    # period is 2 s; 0.5 s elapsed -> 1.5 s left.
    sleep = n._rate_limit_sleep({}, last_update=0.0, now=0.5)
    assert abs(sleep - 1.5) < 1e-9
