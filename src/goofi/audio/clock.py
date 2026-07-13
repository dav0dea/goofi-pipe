import math


class SampleClock:
    """Monotonic sample-count pacer for a rate-`sr` generator.

    `advance(now)` returns how many frames should be emitted so that the
    cumulative emitted count tracks `floor((now - ref) * sr)` without the
    `round(dt * sr)` quantization drift of wall-clock chunk sizing. A single
    oversized gap (long stall) clamps to `max_block` and realigns the reference
    point to `now`, dropping the backlog instead of spending many ticks catching
    up. The realign carries the reference as an exact `(time, emitted)` pair — no
    lossy `now - emitted/sr` roundtrip — so paced ticks stay sample-exact.
    """

    def __init__(self, sr: float, *, max_block: int = 8192, resync_after: float = 0.25) -> None:
        self.sr = float(sr)
        self.max_block = int(max_block)
        self.resync_after = float(resync_after)
        self._ref_time = 0.0
        self._ref_emitted = 0
        self._emitted = 0

    def start(self, now: float) -> None:
        self._ref_time = float(now)
        self._ref_emitted = 0
        self._emitted = 0

    def advance(self, now: float) -> int:
        # Snap the sample product to its integer when within a sub-sample epsilon:
        # wall-clock inputs are floats, so `(now - ref) * sr` carries ~1e-12-sample
        # representation noise that would otherwise floor an exact boundary one short.
        elapsed = (now - self._ref_time) * self.sr
        target = self._ref_emitted + math.floor(elapsed + 1e-6)
        raw = target - self._emitted
        behind = (now - self._ref_time) - (self._emitted - self._ref_emitted) / self.sr
        if raw > self.max_block or behind > self.resync_after:
            # Long stall / oversized backlog: emit at most one block and realign
            # the reference to `now` so the clock resumes paced next tick.
            n = min(max(raw, 0), self.max_block)
            self._emitted += n
            self._ref_time = float(now)
            self._ref_emitted = self._emitted
            return n
        n = raw if raw > 0 else 0
        self._emitted += n
        return n

    @property
    def emitted(self) -> int:
        return self._emitted
