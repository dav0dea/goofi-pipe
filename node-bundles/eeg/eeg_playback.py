"""EEGPlayback — replays a recording as a live stream, through mne's `read_raw`.

Any format mne reads: FIF, EDF, BDF, BrainVision, and the rest. Each tick emits the samples the
wall clock has passed since the last one, so the stream runs at the recording's own rate. The
file is read when its path changes, so a new path needs no restart.
"""

import time

import mne
import numpy as np
import goofi


class EEGPlayback(goofi.Node):
    """Replay an EEG recording at its own rate, as if it were live."""

    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PRODUCER = True
    PARAMS = {
        "playback": {
            "file": goofi.StringParam("", doc="A recording in any format mne reads."),
            "loop": goofi.BoolParam(True, doc="Start over at the end, or stop there."),
            "scale": goofi.FloatParam(1e6, 0.0, 1e9, doc="Multiplier on mne's volts; 1e6 reads as microvolts."),
        }
    }

    def setup(self):
        self.loaded = None
        self.data = None

    def process(self):
        p = self.params.playback
        if p.file != self.loaded:
            self.loaded, self.data = p.file, None
            if p.file:
                self.load(p.file)
        if self.data is None:
            return None

        now = time.monotonic()
        n = int((now - self.last) * self.sfreq)
        if n <= 0:
            return None
        # A stall emits one second and forgets the rest, never a burst of the whole backlog.
        if n > self.sfreq:
            n, self.last = int(self.sfreq), now
        else:
            self.last += n / self.sfreq

        total = self.data.shape[1]
        if p.loop:
            idx = np.arange(self.cursor, self.cursor + n) % total
            self.cursor = (self.cursor + n) % total
        else:
            if self.cursor >= total:
                return None
            idx = np.arange(self.cursor, min(self.cursor + n, total))
            self.cursor += n
        return self.data[:, idx] * p.scale, {"sfreq": self.sfreq, "channels": {"dim0": self.channels}}

    def load(self, path):
        raw = mne.io.read_raw(path, preload=True, verbose=False)
        self.data = raw.get_data().astype(np.float32)
        self.sfreq = float(raw.info["sfreq"])
        self.channels = list(raw.ch_names)
        self.cursor = 0
        self.last = time.monotonic()
