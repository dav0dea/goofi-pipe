"""Rolling per-node execution statistics (src/goofi/node_stats.py)."""
import time

from goofi.node_stats import ExecStats

from .test_manager import _bare_manager


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


# ---- end-to-end: a real spawned node pushes NODE_STATS to its NodeRef --------

def test_live_node_pushes_stats_to_noderef():
    # The node computes stats on its own thread and emits NODE_STATS (~1 Hz) on
    # the status plane; the NodeRef caches the latest snapshot. Drive a real
    # process and wait for the first push to land.
    mgr = _bare_manager(use_multiprocessing=True)
    try:
        a = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[a]
        assert ref.wait_for_state(timeout=2.0)

        # The first push can fire at tick 1 (no rate yet); wait for a steady-state
        # snapshot once enough ticks have accumulated for a rate (~1 push later).
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            stats = ref.node_stats
            if stats is not None and stats["updates_per_second"] > 0:
                break
            time.sleep(0.05)

        stats = ref.node_stats
        assert stats is not None, "node never pushed NODE_STATS"
        assert set(stats) == {"updates_per_second", "mean_process_ms", "total_ticks"}
        assert stats["total_ticks"] > 0
        assert stats["updates_per_second"] > 0  # the Oscillator ticks continuously
    finally:
        mgr.terminate(notify_gui=False)
