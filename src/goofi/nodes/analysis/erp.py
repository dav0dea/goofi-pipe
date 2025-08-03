import time

import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, StringParam


class ERP(Node):
    """
    This node computes the event-related potential (ERP) by averaging segments of input signals that are time-locked to external triggers. Each time a trigger is received, it collects a segment of the signal of fixed duration following that trigger and averages these segments in real time to update the ERP. The result is an averaged response to repeated events or stimuli, useful in signal processing applications such as EEG analysis.

    Inputs:
    - signal: Incoming array data representing the continuous time-series signal to be analyzed and averaged.
    - trigger: Array indicating the presence of a trigger event, used to segment the signal for ERP extraction.

    Outputs:
    - erp: The current averaged event-related potential as an array, along with updated metadata.
    """

    def config_input_slots():
        return {"signal": DataType.ARRAY, "trigger": DataType.ARRAY}

    def config_output_slots():
        return {"erp": DataType.ARRAY}

    def config_params():
        return {
            "erp": {
                "duration": FloatParam(1.0, vmin=0.0, vmax=3.0, doc="Duration of the ERP in seconds"),
                "baseline": StringParam("mean", options=["none", "mean"], doc="Baseline correction method"),
                "baseline_axis": -1,
                "reset": BoolParam(False, trigger=True, doc="Reset average evoked response"),
                "delay_retrigger": BoolParam(True, doc="Wait for length of signal before retriggering ERP"),
            }
        }

    def setup(self):
        self.collecting = None
        self.erp = None
        self.n_samples = 0
        self.last_trigger = None

    def process(self, signal, trigger):
        if self.params.erp.reset.value:
            self.erp = None
            self.collecting = None

        if signal is None:
            return None

        triggered = trigger is not None and not np.any(np.isnan(trigger.data))
        if triggered and self.params.erp.delay_retrigger.value and self.collecting is not None:
            # ignore trigger as we are still collecting data for the last trigger
            triggered = False

        if triggered:
            # new trigger, start collecting data
            self.collecting = []
        elif self.collecting is None:
            # no trigger, not collecting data
            return None

        self.collecting.append(signal.data)

        target_samples = self.params.erp.duration.value * signal.meta["sfreq"]
        if sum(chunk.shape[self.params.erp.baseline_axis.value] for chunk in self.collecting) >= target_samples:
            # enough samples collected, add to ERP
            data = np.concatenate(self.collecting, axis=self.params.erp.baseline_axis.value)
            data = np.take(data, np.arange(target_samples, dtype=int), axis=self.params.erp.baseline_axis.value)

            # baseline correction
            if self.params.erp.baseline.value == "mean":
                data = data - data.mean(axis=self.params.erp.baseline_axis.value, keepdims=True)

            if self.erp is None:
                # first ERP
                self.erp = data
                self.n_samples = 1
            elif self.erp.shape != data.shape:
                raise ValueError(f"Data shape ({data.shape}) changed, ERP has shape {self.erp.shape}.")
            else:
                # add new ERP to average
                self.erp = self.erp + data
                self.n_samples += 1

            self.last_trigger = time.time()
            self.collecting = None

            meta = signal.meta.copy()
            meta["erp_samples"] = self.n_samples
            return {"erp": (self.erp / self.n_samples, meta)}
        else:
            return None

    def erp_duration_changed(self):
        self.erp = None
        self.collecting = None
