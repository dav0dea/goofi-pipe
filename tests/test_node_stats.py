"""Rolling per-node execution statistics (src/goofi/node_stats.py)."""
from goofi.node_stats import ExecStats


def test_empty_stats_are_zeroed():
    s = ExecStats(window=10)
    snap = s.snapshot()
    assert snap["updates_per_second"] == 0.0
    assert snap["mean_process_ms"] == 0.0
    assert snap["total_ticks"] == 0


def test_updates_per_second_from_tick_cadence():
    s = ExecStats(window=10)
    # 11 ticks, 0.1s apart -> span 1.0s, 10 intervals -> 10 updates/s
    for i in range(11):
        s.record(duration=0.0, now=i * 0.1)
    assert s.snapshot()["updates_per_second"] == 10.0
    assert s.snapshot()["total_ticks"] == 11


def test_mean_process_ms_is_windowed():
    s = ExecStats(window=5)
    # durations in seconds; mean over the last 5 -> ms
    for i, d in enumerate([0.001, 0.002, 0.003, 0.004, 0.005]):
        s.record(duration=d, now=i * 0.1)
    assert s.snapshot()["mean_process_ms"] == 3.0  # mean(1..5 ms)


def test_window_drops_old_samples():
    s = ExecStats(window=3)
    # feed 6 ticks; only the last 3 durations count toward the mean, but total
    # keeps the lifetime count
    for i, d in enumerate([0.010, 0.010, 0.010, 0.001, 0.001, 0.001]):
        s.record(duration=d, now=i * 0.1)
    assert s.snapshot()["mean_process_ms"] == 1.0  # last 3 are 1ms each
    assert s.snapshot()["total_ticks"] == 6


def test_single_tick_has_no_rate_yet():
    s = ExecStats(window=10)
    s.record(duration=0.002, now=5.0)
    snap = s.snapshot()
    assert snap["updates_per_second"] == 0.0  # need >=2 timestamps for a rate
    assert snap["mean_process_ms"] == 2.0
    assert snap["total_ticks"] == 1
