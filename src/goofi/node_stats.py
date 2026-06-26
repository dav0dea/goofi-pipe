"""Rolling per-node execution statistics.

Cheap telemetry the node computes about its own ``process()`` cadence — average
update rate and mean process duration over the last N ticks — for display in the
frontend. Updated once per tick on the NODE thread and read on that same thread
when the node emits a NODE_STATS push, so it needs no locking.
"""
from collections import deque


class ExecStats:
    """Windowed stats over the last ``window`` ``process()`` calls."""

    def __init__(self, window: int = 10):
        self._durations = deque(maxlen=window)  # process() seconds, last `window`
        self._timestamps = deque(maxlen=window)  # perf-counter time of each tick
        self._total = 0

    def record(self, duration: float, now: float) -> None:
        """Record one completed ``process()`` call: its ``duration`` (seconds) and
        the time ``now`` (a monotonic perf-counter reading) it finished at."""
        self._durations.append(duration)
        self._timestamps.append(now)
        self._total += 1

    def snapshot(self) -> dict:
        """A small JSON-friendly stats dict (the NODE_STATS payload)."""
        durs = self._durations
        ts = self._timestamps
        mean_ms = (sum(durs) / len(durs) * 1000.0) if durs else 0.0
        ups = 0.0
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            if span > 0:
                ups = (len(ts) - 1) / span
        return {
            "updates_per_second": round(ups, 2),
            "mean_process_ms": round(mean_ms, 3),
            "total_ticks": self._total,
        }
