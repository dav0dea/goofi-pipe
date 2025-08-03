from typing import Any, Dict, Tuple

import numpy as np

from goofi.node import Node
from goofi.params import BoolParam, FloatParam


class EEGRecording(Node):
    """
    Streams EEG recordings as an LSL (Lab Streaming Layer) stream, either from an example dataset, a user-provided file, or a supported MNE-compatible format. This node manages reading, looping, and live replay of EEG data but does not process or modify incoming data.

    Inputs:

    Outputs:
    """

    def config_params():
        return {
            "recording": {
                "use_example_data": True,
                "file_path": "",
                "file_sfreq": FloatParam(256, vmax=1000),
                "source_name": "goofi",
                "stream_name": "recording",
                "loop": BoolParam(True, doc="Whether to loop the recording"),
                "reset": BoolParam(trigger=True, doc="Restart the recording stream"),
            }
        }

    def setup(self):
        """
        Load the data and start the stream.
        """
        import mne
        import pandas as pd
        from mne.datasets import eegbci
        from mne_lsl.player import PlayerLSL

        # stop previous stream if it exists
        self.stop()

        if self.params.recording.use_example_data.value:
            if self.params.recording.file_path.value != "":
                # both use_example_data and file_path are set
                # TODO: add proper logging
                print(
                    f"Both 'use_example_data' and 'file_path' are set. Prioritizing file: {self.params.recording.file_path.value}"
                )

        assert self.params.recording.stream_name.value != "", "Stream name cannot be empty."

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
                [mne.io.read_raw(p, preload=True, verbose=False) for p in eegbci.load_data(1, [1, 2], update_path=False)],
                verbose=False,
            )
            eegbci.standardize(raw)
            # scale the data for better default behavior
            raw.apply_function(lambda x: x * 1e4)
        else:
            raise RuntimeError("No data source specified. Set either 'use_example_data' or 'file_path'.")

        # start the stream
        self.stream = PlayerLSL(
            raw,
            name=self.params.recording.stream_name.value,
            source_id=self.params.recording.source_name.value,
            n_repeat=np.inf if self.params.recording.loop.value else 1,
            annotations=False,
        )
        self.stream.start()

        # NOTE: this is a special case since the node doesn't process data, so errors are never cleared
        self.clear_error()

    def stop(self):
        """Stop the stream if it exists."""
        if hasattr(self, "stream") and self.stream is not None:
            self.stream.stop()
            self.stream = None

    def process(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """Should never run, as the stream runs on its own."""
        raise NotImplementedError

    def recording_use_example_data_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_file_path_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_file_sfreq_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_source_name_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_stream_name_changed(self, _):
        """Reinitialize the stream."""
        self.setup()

    def recording_loop_changed(self, value: bool):
        """Set the loop parameter of the stream."""
        # ideally we wouldn't reset the stream, not sure if there is a way to change the n_repeat parameter on the fly
        self.setup()

    def recording_reset_changed(self, value: bool):
        """Reset the stream if the reset parameter is triggered."""
        self.setup()

    def terminate(self):
        """Stop the stream and terminate the node."""
        self.stop()
