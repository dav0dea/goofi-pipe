import numpy as np


class AudioRing:
    """Bounded SPSC float32 ring buffer, block shape `(frames, channels)`.

    Single producer (`write`) / single consumer (`read_into`). Overflow is
    reject-newest: a write only accepts what currently fits and drops the
    remainder (bumping `overruns` once). A short read copies what is available
    and leaves the rest of `out` untouched (bumping `underruns` once); the
    caller zero-fills. One preallocated buffer, no per-call allocation.

    Lock-free: the producer owns `_w` (total frames written), the consumer owns
    `_r` (total frames read); both increase monotonically and occupancy is
    DERIVED as `_w - _r`. No field is written by both threads, so there is no
    shared read-modify-write whose update could be lost — correct even without
    the GIL. (`_buf` is indexed modulo capacity; the counters themselves never
    wrap, and Python ints are unbounded.)
    """

    def __init__(self, capacity_frames: int, channels: int) -> None:
        self._cap = int(capacity_frames)
        self._ch = int(channels)
        self._buf = np.zeros((self._cap, self._ch), dtype=np.float32)
        self._w = 0  # producer-owned, monotonic
        self._r = 0  # consumer-owned, monotonic
        self.overruns = 0
        self.underruns = 0

    def write(self, block: np.ndarray) -> int:
        n = block.shape[0]
        take = min(n, self._cap - (self._w - self._r))  # single LOAD of consumer's _r
        if take < n:
            self.overruns += 1
        if take:
            start = self._w % self._cap
            end = start + take
            if end <= self._cap:
                self._buf[start:end] = block[:take]
            else:
                first = self._cap - start
                self._buf[start:] = block[:first]
                self._buf[: end - self._cap] = block[first:take]
            self._w += take  # publish only after the payload is in place
        return take

    def read_into(self, out: np.ndarray) -> int:
        n = out.shape[0]
        take = min(self._w - self._r, n)  # single LOAD of producer's _w
        if take:
            start = self._r % self._cap
            end = start + take
            if end <= self._cap:
                out[:take] = self._buf[start:end]
            else:
                first = self._cap - start
                out[:first] = self._buf[start:]
                out[first:take] = self._buf[: end - self._cap]
            self._r += take
        if take < n:
            self.underruns += 1
        return take

    @property
    def fill(self) -> int:
        return self._w - self._r

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def channels(self) -> int:
        return self._ch
