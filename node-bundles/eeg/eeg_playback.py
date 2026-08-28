"""EegPlayback — replays a recording as a live stream, through mne's `read_raw`.

Any format mne reads: FIF, EDF, BDF, BrainVision, and the rest. Each tick emits the samples the
wall clock has passed since the last one, so the stream runs at the recording's own rate. The
file is read when its path changes, so a new path needs no restart. By default `file` follows an
expression into goofi's own sample folder, and a sample not there yet downloads on a background
thread — the node is silent until it lands, then plays.
"""

import os
import shutil
import threading
import time
import urllib.request

import mne
import numpy as np
import goofi

SAMPLES = {
    # A sample is a LIST of files, named file last, so a half-landed set never looks loadable.
    # 31 MB: eyes-closed resting EEG from the SRM resting-state dataset (OpenNeuro ds003775).
    "eeg-rest-srm.edf": [
        "https://s3.amazonaws.com/openneuro.org/ds003775/sub-001/ses-t1/eeg/sub-001_ses-t1_task-resteyesc_eeg.edf",
    ],
    # 105 MB: one full-night sleep PSG from PhysioNet's HMC sleep staging set.
    "eeg-sleep-hmc.edf": [
        "https://physionet.org/files/hmc-sleep-staging/1.1/recordings/SN001.edf",
    ],
}


class EegPlayback(goofi.Node):
    """Replay an EEG recording at its own rate, as if it were live."""

    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PRODUCER = True
    PARAMS = {
        "playback": {
            "sample": goofi.StringParam(
                "eeg-rest-srm.edf",
                list(SAMPLES),
                doc="Sample recordings the default `file` expression follows; each downloads on first use.",
            ),
            "file": goofi.StringParam(
                "",
                doc="A recording in any format mne reads.",
                expression='globals.goofi_home + "/data/samples/" + me.params.playback.sample',
            ),
            "loop": goofi.BoolParam(True, doc="Start over at the end, or stop there."),
            "scale": goofi.FloatParam(1e6, 0.0, 1e9, doc="Multiplier on mne's volts; 1e6 reads as microvolts."),
        }
    }

    def setup(self):
        self.loaded = None
        self.data = None
        self.fetching = None
        self.failed = None

    def process(self):
        p = self.params.playback
        if self.failed is not None:
            err, self.failed = self.failed, None
            raise err
        if p.file != self.loaded:
            self.loaded, self.data = p.file, None
            if p.file:
                if os.path.exists(p.file) or os.path.basename(p.file) not in SAMPLES:
                    self.load(p.file)
                else:
                    self.fetch(p.file)
        if self.data is None:
            if self.fetching is not None and not self.fetching.is_alive():
                # The download ended; forgetting `loaded` sends the next tick back through the load.
                self.fetching = None
                self.loaded = None
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

    def fetch(self, target):
        def run():
            try:
                download(SAMPLES[os.path.basename(target)], target)
            except Exception as e:
                self.failed = e

        self.fetching = threading.Thread(target=run, daemon=True, name="eeg-sample-download")
        self.fetching.start()


def download(urls, target):
    folder = os.path.dirname(target)
    if folder:
        os.makedirs(folder, exist_ok=True)
    stem, _ = os.path.splitext(target)
    for url in urls:
        dest = stem + os.path.splitext(url)[1]
        if os.path.exists(dest):
            continue
        part = dest + ".part"
        with urllib.request.urlopen(url, timeout=60) as body, open(part, "wb") as out:
            shutil.copyfileobj(body, out)
        os.replace(part, dest)
