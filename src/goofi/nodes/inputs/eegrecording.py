import logging
import time
from typing import Any, Dict, Tuple

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam

logger = logging.getLogger(__name__)


class EEGRecording(Node):
    """
    Replays an EEG recording (example dataset, a user-provided file, or any
    MNE-compatible format) as a direct data stream. Each tick emits the next
    slice of samples on the `out` slot, paced to the recording's sampling rate,
    looping when enabled.

    Inputs:

    Outputs:
    - out: The next chunk of recording samples (channels x samples), with
      metadata carrying the sampling frequency and channel names.
    """

    def config_params():
        return {
            "recording": {
                "use_example_data": True,
                "example_data_subject": 1,
                "file_path": "",
                "file_sfreq": FloatParam(256, vmax=1000),
                "loop": BoolParam(True, doc="Whether to loop the recording"),
                "reset": BoolParam(trigger=True, doc="Restart the recording from the beginning"),
            },
            "common": {"autotrigger": True, "max_frequency": 30},
        }

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        """Load the recording into memory and reset the read cursor."""
        if self.params.recording.use_example_data.value and self.params.recording.file_path.value != "":
            # both use_example_data and file_path are set
            logger.warning(
                "Both 'use_example_data' and 'file_path' are set. Prioritizing file: %s",
                self.params.recording.file_path.value,
            )

        if self.params.recording.file_path.value != "":
            if self.params.recording.file_path.value.endswith(".csv"):
                # load data from csv file
                df = pd.read_csv(self.params.recording.file_path.value, index_col=0)
                df = df.select_dtypes(include=["float"])
                data = df.transpose().to_numpy()

                sfreq = self.params.recording.file_sfreq.value
                sfreq = sfreq if sfreq > 0 else 256
                info = mne.create_info(ch_names=df.columns.tolist(), ch_types=["eeg"] * data.shape[0], sfreq=sfreq)
                raw = mne.io.RawArray(data, info)
            else:
                # load data from an MNE-compatible file
                raw = mne.io.read_raw(self.params.recording.file_path.value, preload=True)
        elif self.params.recording.use_example_data.value:
            raw = mne.concatenate_raws(
                [
                    mne.io.read_raw(p, preload=True, verbose=False)
                    for p in eegbci.load_data(
                        self.params.recording.example_data_subject.value, [1, 2], update_path=False
                    )
                ],
                verbose=False,
            )
            eegbci.standardize(raw)
            # scale the data for better default behavior
            raw.apply_function(lambda x: x * 1e4)
        else:
            raise RuntimeError("No data source specified. Set either 'use_example_data' or 'file_path'.")

        # cache the recording as a plain array + its metadata, and reset the cursor
        self.data = raw.get_data()
        self.sfreq = float(raw.info["sfreq"])
        self.ch_names = list(raw.ch_names)
        self.idx = 0
        self.last_trigger = time.time()
        self.clear_error()

    @staticmethod
    def _read_chunk(data, idx, n, loop):
        """Read `n` samples starting at column `idx` from `data` (channels × total).

        loop=True wraps around (valid for any n, even n > total); loop=False
        clamps to the end and returns (None, total) once idx is at the end, so
        the node idles until reset."""
        total = data.shape[1]
        if loop:
            return data[:, np.arange(idx, idx + n) % total], (idx + n) % total
        if idx >= total:
            return None, total
        end = min(idx + n, total)
        return data[:, idx:end], end

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """Emit the samples that fall within the wall-clock time since the last
        tick — round(dt * sfreq) of them — advancing the cursor (looping if
        enabled). A late tick emits one larger (but bounded) chunk, never a
        sustained burst."""
        t = time.time()
        n = int(np.round((t - self.last_trigger) * self.sfreq))
        self.last_trigger = t
        if n <= 0:
            return None

        chunk, self.idx = self._read_chunk(self.data, self.idx, n, self.params.recording.loop.value)
        if chunk is None:
            return None

        return {"out": (chunk, {"sfreq": self.sfreq, "channels": {"dim0": self.ch_names}})}

    def recording_use_example_data_changed(self, _):
        """Reload the recording."""
        self.setup()

    def recording_example_data_subject_changed(self, _):
        """Reload the recording."""
        self.setup()

    def recording_file_path_changed(self, _):
        """Reload the recording."""
        self.setup()

    def recording_file_sfreq_changed(self, _):
        """Reload the recording."""
        self.setup()

    def recording_reset_changed(self, value: bool):
        """Restart the recording from the beginning."""
        self.idx = 0
        self.last_trigger = time.time()
