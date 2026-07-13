"""The processing loop's max_frequency throttle — one universal knob.

max_frequency <= 0 means unbounded: the tick runs as fast as the loop is driven.
A free-running producer (autotrigger) sets a finite cap so it can't busy-spin;
an input-triggered consumer leaves it unbounded (the default) so it runs at its
producer's rate. No per-node / per-delivery-mode special-casing."""
from goofi.data import DataType
from tests.utils import make_custom_node


def _node():
    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY})
    return Cls.create_standalone()


def test_unbounded_when_cap_is_zero():
    n = _node()
    n.params.common.max_frequency.value = 0.0
    assert n._rate_limit_sleep(last_update=0.0, now=0.0) is None


def test_unbounded_when_cap_is_negative():
    n = _node()
    n.params.common.max_frequency.value = -1.0
    assert n._rate_limit_sleep(last_update=0.0, now=0.0) is None


def test_normal_tick_sleeps_the_remaining_period():
    n = _node()
    n.params.common.max_frequency.value = 30.0  # updates-per-second -> 1/30 s period
    # 10 ms into a 33.3 ms period -> ~23.3 ms left to sleep.
    sleep = n._rate_limit_sleep(last_update=0.0, now=0.010)
    assert abs(sleep - (1.0 / 30.0 - 0.010)) < 1e-9


def test_behind_schedule_returns_non_positive_no_sleep():
    n = _node()
    n.params.common.max_frequency.value = 30.0
    # Already 100 ms past a 33.3 ms period -> negative -> caller won't sleep.
    assert n._rate_limit_sleep(last_update=0.0, now=0.100) <= 0.0


def test_seconds_per_update_mode_uses_period_directly():
    n = _node()
    n.params.common.max_frequency.value = 2.0
    n.params.common.frequency_mode.value = "seconds-per-update"
    # period is 2 s; 0.5 s elapsed -> 1.5 s left.
    sleep = n._rate_limit_sleep(last_update=0.0, now=0.5)
    assert abs(sleep - 1.5) < 1e-9
