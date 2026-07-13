import time

import numpy as np

from goofi.audio.clock import SampleClock
from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


class Oscillator(Node):
    """
    This node generates basic oscillator waveforms such as sine, square, sawtooth, and pulse at a specified frequency and sampling rate. It produces a continuous stream of samples corresponding to the selected waveform. The frequency of the oscillator can be controlled in real-time by providing an input array; otherwise, a set frequency parameter is used.

    Inputs:
    - frequency: Optional. An array containing the frequency (Hz) with which to generate samples. If not provided, a default frequency is used.

    Outputs:
    - out: An array of generated waveform samples, along with metadata that includes the sampling frequency.
    """

    def config_params():
        return {
            "oscillator": {
                "type": StringParam("sine", options=["sine", "square", "sawtooth", "pulse"]),
                "frequency": FloatParam(440.0, 0.1, 20000.0),
                "sampling_frequency": FloatParam(48000.0, 1.0, 192000.0),
            },
            "square": {"duty_cycle": FloatParam(0.5, 0.0, 1.0)},
            # Free-run, paced by the tick rate limiter so the loop can't busy-spin.
            # The SampleClock still owns the EXACT sample count per tick (no round(dt)
            # quantization, no drift) — the limiter only sets how often we wake and
            # thus the audio block granularity (200 Hz -> ~5 ms blocks at 48 kHz).
            "common": {"autotrigger": True, "max_frequency": FloatParam(200.0, 0.0, 1000.0)},
        }

    def config_input_slots():
        return {"frequency": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def setup(self):
        self.phase = 0.0
        self.clock = SampleClock(self.params.oscillator.sampling_frequency.value)
        self.clock.start(time.time())

    def oscillator_sampling_frequency_changed(self, value):
        # Re-anchor the pacer to the live rate so it stops emitting the old
        # samples/sec while process() has already switched to the new one (a
        # detune/timebase mismatch otherwise). Re-anchor at `now` to avoid a
        # backlog jump; leave self.phase so the waveform stays phase-continuous.
        self.clock = SampleClock(value)
        self.clock.start(time.time())

    def process(self, frequency):
        freq = frequency.data.item() if frequency else self.params.oscillator.frequency.value
        sfreq = self.params.oscillator.sampling_frequency.value
        osc_type = self.params.oscillator.type.value

        meta = {"sfreq": sfreq}

        # The SampleClock owns pacing: emit exactly the samples that have elapsed.
        n = self.clock.advance(time.time())
        if n == 0:
            return {"out": (np.array([]), meta)}

        inc = 2 * np.pi * freq / sfreq
        i = np.arange(n)
        raw = self.phase + inc * i          # unwrapped phase at each sample's start
        osc_phase = raw % (2 * np.pi)        # == the per-sample old_phase of the old loop

        if osc_type == "sine":
            data = np.sin(osc_phase)
        elif osc_type == "square":
            duty_cycle = self.params.square.duty_cycle.value
            data = np.where(osc_phase < 2 * np.pi * duty_cycle, 1.0, -1.0)
        elif osc_type == "sawtooth":
            data = osc_phase / np.pi - 1.0
        elif osc_type == "pulse":
            # Fire on the sample whose step crosses a 2*pi boundary (phase wrap).
            next_raw = raw + inc
            data = (np.floor(next_raw / (2 * np.pi)) > np.floor(raw / (2 * np.pi))).astype(float)
        else:
            data = np.zeros(n)

        # Carry the phase forward (mod 2*pi) so the next block continues seamlessly.
        self.phase = float((self.phase + inc * n) % (2 * np.pi))

        return {"out": (data, meta)}
