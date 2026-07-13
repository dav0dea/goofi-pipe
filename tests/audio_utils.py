"""Hardware-free stand-ins for the audio Phase-5 tests.

`FakeStream` replaces `sd.OutputStream`: it captures the injected callback and
lets a test drive the DAC clock deterministically with `pump(frames)`, exactly
as PortAudio would. `FixedClock` replaces `SampleClock` so a paced generator can
be driven with scripted block sizes, free of wall-clock timing.
"""
import numpy as np


class FakeStream:
    def __init__(self, *, samplerate, channels, blocksize, callback, device=None):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.callback = callback
        self.device = device
        self.started = False
        self.closed = False
        # Everything the callback wrote to `outdata`, one (frames, channels) chunk per pump.
        self.captured = []

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def pump(self, frames):
        """Invoke the stored callback for `frames` samples, as the backend would."""
        out = np.zeros((frames, self.channels), dtype=np.float32)
        self.callback(out, frames, None, None)
        self.captured.append(out.copy())
        return out


def fake_stream_factory(**kwargs):
    """A `stream_factory` that hands the node a `FakeStream` instead of a device."""
    return FakeStream(**kwargs)


class FixedClock:
    """A `SampleClock` stand-in that emits a scripted sequence of block sizes."""

    def __init__(self, sizes):
        self._sizes = list(sizes)

    def start(self, now):
        pass

    def advance(self, now):
        return self._sizes.pop(0) if self._sizes else 0

    @property
    def emitted(self):
        return 0
