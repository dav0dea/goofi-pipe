import numpy as np


class AudioRing:
    """Bounded SPSC float32 ring buffer, block shape `(frames, channels)`.

    Single producer (`write`) / single consumer (`read_into`). Overflow is
    reject-newest: a write only accepts what currently fits and drops the
    remainder (bumping `overruns` once). A short read copies what is available
    and leaves the rest of `out` untouched (bumping `underruns` once); the
    caller zero-fills. One preallocated buffer, no per-call allocation.
    """

    def __init__(self, capacity_frames: int, channels: int) -> None:
        self._cap = int(capacity_frames)
        self._ch = int(channels)
        self._buf = np.zeros((self._cap, self._ch), dtype=np.float32)
        self._w = 0
        self._r = 0
        self._fill = 0
        self.overruns = 0
        self.underruns = 0

    def write(self, block: np.ndarray) -> int:
        n = block.shape[0]
        take = min(n, self._cap - self._fill)
        if take < n:
            self.overruns += 1
        if take:
            end = self._w + take
            if end <= self._cap:
                self._buf[self._w:end] = block[:take]
            else:
                first = self._cap - self._w
                self._buf[self._w:] = block[:first]
                self._buf[: end - self._cap] = block[first:take]
            self._w = end % self._cap
            self._fill += take
        return take

    def read_into(self, out: np.ndarray) -> int:
        n = out.shape[0]
        take = min(self._fill, n)
        if take:
            end = self._r + take
            if end <= self._cap:
                out[:take] = self._buf[self._r:end]
            else:
                first = self._cap - self._r
                out[:first] = self._buf[self._r:]
                out[first:take] = self._buf[: end - self._cap]
            self._r = end % self._cap
            self._fill -= take
        if take < n:
            self.underruns += 1
        return take

    @property
    def fill(self) -> int:
        return self._fill

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def channels(self) -> int:
        return self._ch
