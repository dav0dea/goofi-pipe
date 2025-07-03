import time

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class Oscillator(Node):
    def config_params():
        return {
            "oscillator": {
                "type": StringParam("sine", options=["sine", "square", "sawtooth", "pulse"]),
                "frequency": FloatParam(1.0, 0.1, 30.0),
                "sampling_frequency": FloatParam(1000.0, 1.0, 1000.0),
            },
            "square": {"duty_cycle": FloatParam(0.5, 0.0, 1.0)},
            "common": {"autotrigger": True},
        }

    def config_input_slots():
        return {"frequency": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        self.phase = 0.0
        self.last_trigger = time.time()

    def process(self, frequency):
        freq = frequency.data.item() if frequency else self.params.oscillator.frequency.value
        sfreq = self.params.oscillator.sampling_frequency.value
        osc_type = self.params.oscillator.type.value

        meta = {"sfreq": sfreq}

        t = time.time()
        dt = t - self.last_trigger
        n_samples = int(np.round(dt * sfreq))
        self.last_trigger = t

        if n_samples == 0:
            return {"out": (np.array([]), meta)}

        data = np.zeros(n_samples)
        phase_increment = 2 * np.pi * freq / sfreq

        for i in range(n_samples):
            old_phase = self.phase
            self.phase += phase_increment
            if self.phase >= 2 * np.pi:
                self.phase -= 2 * np.pi

            if osc_type == "sine":
                data[i] = np.sin(old_phase)
            elif osc_type == "square":
                duty_cycle = self.params.square.duty_cycle.value
                data[i] = 1.0 if old_phase < 2 * np.pi * duty_cycle else -1.0
            elif osc_type == "sawtooth":
                data[i] = old_phase / np.pi - 1.0
            elif osc_type == "pulse":
                if self.phase < old_phase:  # phase wrapped
                    data[i] = 1.0
                else:
                    data[i] = 0.0

        return {"out": (data, meta)}
