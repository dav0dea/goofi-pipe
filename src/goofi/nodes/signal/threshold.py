import time

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, StringParam


class Threshold(Node):
    """
    Applies a thresholding operation to an input array. The node compares each value in the input array to a specified threshold using one of several modes (e.g., greater than, less than), and outputs an array where each element is set to a true or false value depending on the comparison result. The node supports features such as minimum delay between triggers, retrigger requirements, and outputting NaN after certain threshold crossings.

    Inputs:
    - data: The input array to be thresholded.

    Outputs:
    - thresholded: The thresholded array with elements set according to whether the input values passed the threshold condition.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"thresholded": DataType.ARRAY}

    def config_params():
        return {
            "threshold": {
                "threshold": FloatParam(0.0, -10, 10, doc="Threshold value"),
                "mode": StringParam(">", options=[">", ">=", "<", "<="]),
                "true_value": FloatParam(1.0, doc="Value to send when threshold is exceeded"),
                "false_value": FloatParam(0.0, doc="Value to send when threshold is not exceeded"),
                "trigger_on_false": BoolParam(False, doc="Send data even when threshold is not exceeded"),
                "require_pass": BoolParam(False, doc="Only retrigger if the threshold was not exceeded in the last cycle"),
                "min_delay": FloatParam(0.0, doc="Minimum delay between two threshold crossings in seconds"),
                "nan_reset": BoolParam(False, doc="Send NaN in the first cycle after a threshold crossing"),
            },
        }

    def setup(self):
        self.last_trigger = 0
        self.triggered_last_cycle = False
        self.trigger_ready = True

    def process(self, data: Data):
        if self.triggered_last_cycle and self.params.threshold.nan_reset.value:
            self.triggered_last_cycle = False
            return {"thresholded": (np.array([np.nan]), data.meta)}

        if data is None or data.data is None:
            return None

        if self.params.threshold.mode.value == ">":
            thresholded = data.data > self.params.threshold.threshold.value
        elif self.params.threshold.mode.value == ">=":
            thresholded = data.data >= self.params.threshold.threshold.value
        elif self.params.threshold.mode.value == "<":
            thresholded = data.data < self.params.threshold.threshold.value
        elif self.params.threshold.mode.value == "<=":
            thresholded = data.data <= self.params.threshold.threshold.value

        if not np.any(thresholded):
            # no threshold exceeded
            self.trigger_ready = True

            if not self.params.threshold.trigger_on_false.value:
                # no threshold exceeded and trigger_on_false is False
                return None

        if time.time() - self.last_trigger < self.params.threshold.min_delay.value:
            # threshold exceeded but min_delay not reached
            return None

        if self.params.threshold.require_pass.value and not self.trigger_ready:
            # threshold exceeded but require_pass is True and threshold was already exceeded in the last cycle
            return None

        if np.any(thresholded):
            # threshold exceeded
            self.trigger_ready = False

        self.last_trigger = time.time()
        result = np.where(thresholded, self.params.threshold.true_value.value, self.params.threshold.false_value.value)
        self.triggered_last_cycle = True
        return {"thresholded": (result, data.meta)}
