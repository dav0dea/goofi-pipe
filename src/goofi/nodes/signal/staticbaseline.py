import time

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, IntParam, StringParam


class StaticBaseline(Node):
    """
    This node computes a normalized version of an incoming data array using a baseline period. The baseline is accumulated over a fixed duration, after which normalization is performed using either mean-based z-scoring or quantile transformation. The baseline window is reset and accumulated anew when triggered, allowing normalization to adapt to different periods of the data stream.

    Inputs:
    - data: The 1D or 2D array of data to be normalized.
    - trigger_baseline: Optional input. Triggers a reset of the baseline window when any value in the array is greater than zero.
    - n_seconds: Optional input. Specifies the number of seconds to use for accumulating the baseline window.

    Outputs:
    - normalized: The normalized array using statistics from the accumulated baseline window, matching the shape of the input data.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY, "trigger_baseline": DataType.ARRAY, "n_seconds": DataType.ARRAY}

    def config_output_slots():
        return {"normalized": DataType.ARRAY}

    def config_params():
        return {
            "baseline": {
                "n_seconds": IntParam(30, 1, 120),
                "method": StringParam("quantile", options=["quantile", "mean"]),
                "baseline_computation": BoolParam(trigger=True),
            }
        }

    def setup(self):
        from scipy.stats import rankdata

        self.rankdata = rankdata

        self.window = []
        self.time_origin = None

    def process(self, data: Data, trigger_baseline: Data = None, n_seconds: Data = None):
        if data is None or data.data is None:
            return None

        val = np.asarray(data.data)
        if val.ndim > 2:
            print("Error: StaticBaseline only accepts 1D or 2D arrays")
            return None

        normalized_value = np.zeros_like(val)

        # Handle trigger_baseline input
        if trigger_baseline is not None and trigger_baseline.data is not None:
            if np.any(trigger_baseline.data > 0):  # Trigger if any value > 0
                self.window = []
                self.time_origin = time.time()
            self.input_slots["trigger_baseline"].clear()  # Clear the input slot

        # Handle n_seconds input - use input value if provided, otherwise use parameter
        current_n_seconds = self.params["baseline"]["n_seconds"].value
        if n_seconds is not None and n_seconds.data is not None:
            # Use the input value directly
            input_seconds = float(n_seconds.data.item()) if n_seconds.data.size == 1 else float(n_seconds.data[0])
            current_n_seconds = input_seconds
            self.input_slots["n_seconds"].clear()

        # If baseline computation is triggered via parameter, reset the window and set time_origin
        if self.params["baseline"]["baseline_computation"].value:
            self.window = []
            self.time_origin = time.time()

        if self.time_origin:
            elapsed_time = time.time() - self.time_origin

            if elapsed_time < current_n_seconds:
                self.window.extend(val)
            else:
                self.time_origin = None  # Reset time_origin after accumulating for n_seconds

        # Only compute the Z-score when we have enough data in our window
        if len(self.window) >= current_n_seconds:
            # Compute the Z-score using the desired method
            if self.params["baseline"]["method"].value == "quantile":
                normalized_value = self.quantile_transform(val)
            elif self.params["baseline"]["method"].value == "mean":
                normalized_value = self.zscore(val)
        else:
            normalized_value = val  # You can choose to normalize or just use raw data when the window is not full

        return {"normalized": (normalized_value, data.meta)}

    def zscore(self, val):
        if val.ndim == 1:
            mean = np.mean(self.window)
            std = np.std(self.window)
            return (val - mean) / (std + 1e-8)
        elif val.ndim == 2:
            normalized = np.zeros_like(val)
            for i in range(val.shape[0]):
                mean = np.mean(self.window[i])
                std = np.std(self.window[i])
                normalized[i, :] = (val[i, :] - mean) / (std + 1e-8)
            return normalized

    def quantile_transform(self, val):
        # Check for dimension
        if val.ndim == 2:
            normalized = np.zeros_like(val)
            for i in range(val.shape[0]):
                normalized[i, :] = self._quantile_transform_1D(val[i, :])
            return normalized
        else:
            return self._quantile_transform_1D(val)

    def _quantile_transform_1D(self, arr):
        # Convert data to ranks
        ranks = self.rankdata(arr)
        # Scale ranks to [0, 1]
        scaled_ranks = (ranks - 1) / (len(arr) - 1)
        # Calculate quantile values
        quantiles = np.percentile(arr, 100 * scaled_ranks)
        # Return normalized quantile values for the input data
        return (arr - np.mean(quantiles)) / (np.std(quantiles) + 1e-8)
