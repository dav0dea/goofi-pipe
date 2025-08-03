import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Hilbert(Node):
    """
    Computes the analytic signal of the input data using the Hilbert transform and extracts key instantaneous signal properties. Outputs the instantaneous amplitude, phase, and frequency of the input array, useful for advanced signal processing and time-frequency analysis.

    Inputs:
    - data: Input array representing one or more time-series signals.

    Outputs:
    - inst_amplitude: Instantaneous amplitude of the analytic signal for each input channel.
    - inst_phase: Instantaneous phase of the analytic signal for each input channel.
    - inst_frequency: Instantaneous frequency derived from unwrapped phase differences for each input channel.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"inst_amplitude": DataType.ARRAY, "inst_phase": DataType.ARRAY, "inst_frequency": DataType.ARRAY}

    def setup(self):
        from scipy.signal import hilbert

        self.hilbert = hilbert

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        analytic_signal = self.hilbert(data.data)
        inst_amplitude = np.abs(analytic_signal)
        inst_phase = np.angle(analytic_signal)

        # Compute the instantaneous frequency:
        delta_phase = np.diff(np.unwrap(inst_phase), axis=-1)
        inst_frequency = delta_phase / (2.0 * np.pi)

        # Pad inst_frequency to make it the same length as inst_amplitude and inst_phase.
        # Using padding with the last value to keep the size consistent with inst_amplitude and inst_phase.
        pad_value = inst_frequency[..., -1:] if inst_frequency.ndim > 1 else [inst_frequency[-1]]
        inst_frequency = np.concatenate((inst_frequency, pad_value), axis=-1)

        return {
            "inst_amplitude": (inst_amplitude, {**data.meta}),
            "inst_phase": (inst_phase, {**data.meta}),
            "inst_frequency": (inst_frequency, {**data.meta}),
        }
