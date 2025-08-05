from collections import deque

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Cycle(Node):
    """
    This node computes the average cycle shape of an oscillatory signal. It buffers incoming array data and, once enough data is collected, segments it into cycles based on a specified frequency. The node then averages a specified number of these cycles to create a representative cycle waveform of the input signal. This can be used to analyze the typical shape of periodic signals.

    Inputs:
    - signal: Array data representing the signal to be processed. The input must have a sampling frequency specified in its metadata.

    Outputs:
    - cycle: Array data containing the averaged cycle waveform computed from the input signal. The output has the same dimensionality as the input.
    """

    def config_input_slots():
        return {"signal": DataType.ARRAY}

    def config_output_slots():
        return {"cycle": DataType.ARRAY}

    def config_params():
        return {"cycle": {"frequency": FloatParam(10, 0.1, 200), "axis": -1, "num_average": 10}}

    def setup(self):
        self.buffer = None
        self.sfreq = None
        self.num_added = 0

    def process(self, signal: Data):
        if signal is None:
            return None

        if "sfreq" not in signal.meta:
            raise ValueError("Sampling frequency not found in input signal.")

        if self.sfreq != signal.meta["sfreq"]:
            self.sfreq = signal.meta["sfreq"]
            self.init_buffer()

        chunk = signal.data
        assert chunk.ndim <= 2, "Signal must be 1D or 2D."

        n_per_cycle = int(self.sfreq / self.params.cycle.frequency.value)

        # convert to 2D with time on axis 0
        axis = self.params.cycle.axis.value
        if axis < 0:
            axis = chunk.ndim + axis
        if axis == 1:
            chunk = chunk.T

        # expand buffer
        self.buffer.extend(chunk)
        # keep track of number of added samples
        self.num_added += len(chunk)
        self.num_added = self.num_added % n_per_cycle

        if len(self.buffer) < n_per_cycle * self.params.cycle.num_average.value:
            return None

        # average cycles
        idxs = slice(-n_per_cycle * self.params.cycle.num_average.value)
        if self.num_added > 0:
            idxs = slice(-(n_per_cycle * self.params.cycle.num_average.value + self.num_added), -self.num_added)
        cycles = np.array(self.buffer)[idxs]
        cycles = cycles.reshape(self.params.cycle.num_average.value, n_per_cycle, -1)
        cycles = cycles.mean(axis=0).squeeze()

        if axis == 1:
            cycles = cycles.T

        return {"cycle": (cycles, signal.meta)}

    def init_buffer(self):
        # keep one extra cycle to allow for incomplete cycles
        self.buffer = deque(
            maxlen=int(self.sfreq / self.params.cycle.frequency.value) * (self.params.cycle.num_average.value + 1)
        )
        self.num_added = 0

    def cycle_num_average_changed(self, num_average):
        self.init_buffer()

    def cycle_frequency_changed(self, frequency):
        self.init_buffer()
