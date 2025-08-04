import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class EEGHeadsetDetection(Node):
    """
    This node detects whether an EEG headset is worn or not by analyzing incoming EEG data. It monitors the average value of the EEG signal to determine the status and also handles situations where data is missing to decide if the headset is disconnected.

    Inputs:
    - eeg_data: The EEG signal data provided as an array.

    Outputs:
    - headset_status: An array indicating the detected status of the headset: 0 if disconnected, 1 if not worn, or 2 if worn.
    """

    def config_params():
        return {
            "threshold": {
                "threshold_value": FloatParam(200.0, 0.0, 1000.0, doc="Threshold for average signal value"),
                "no_data_threshold": IntParam(
                    30, 0, 1000, doc="Number of updates without data to assume headset is not connected"
                ),
                "center_axis": IntParam(0, doc="Axis along which to center the data"),
            },
            "common": {"autotrigger": True, "max_frequency": 10},
        }

    def config_input_slots():
        return {"eeg_data": DataType.ARRAY}

    def config_output_slots():
        return {"headset_status": DataType.ARRAY, "centered_abs_data": DataType.ARRAY}

    def setup(self):
        self.no_data_count = 0
        self.last_state = np.array(0)

    def process(self, eeg_data: Data):
        if eeg_data is None or eeg_data.data.size == 0:
            self.no_data_count += 1
            if self.no_data_count >= self.params.threshold.no_data_threshold.value:
                # no data for a while, assume headset is not connected
                self.last_state = np.array(0)
            # no data, return last state
            return {"headset_status": (self.last_state, {}), "centered_abs_data": None}
        else:
            self.no_data_count = 0

        self.input_slots["eeg_data"].clear()

        # Extract EEG data
        signal = eeg_data.data

        # Compute the average of the absolute centered values
        signal = np.abs(signal - np.mean(signal, axis=self.params.threshold.center_axis.value))
        max_signal = np.max(signal)

        # Detection logic based on average value threshold
        if max_signal > self.params.threshold.threshold_value.value:
            headset_worn = np.array(1)  # Not worn (average too high)
        else:
            headset_worn = np.array(2)  # Worn (average below threshold)

        # handle the case where the EEG LSL cuts out
        self.input_slots["eeg_data"].clear()

        self.last_state = headset_worn
        return {"headset_status": (headset_worn, {}), "centered_abs_data": (signal, eeg_data.meta)}
