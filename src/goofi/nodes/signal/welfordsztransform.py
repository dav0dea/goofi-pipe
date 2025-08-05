import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam


class WelfordsZTransform(Node):
    """
    This node performs online z-score normalization of input array data using Welford's algorithm for numerically stable, running computation of mean and standard deviation. It updates normalization statistics incrementally as new values arrive, allowing for continual, real-time signal normalization without buffering the entire input stream. The node handles both 1D and 2D arrays, normalizing each individual signal channel independently. Initially, output values are set to zero until sufficient samples have been collected for reliable statistics. Strong outliers are clipped to within a configurable range.

    Inputs:
    - data: 1D or 2D array of numerical values to be normalized in real time.

    Outputs:
    - normalized: Array of the same shape as input, containing the online z-score normalized values.
    """

    INIT_STEPS = 50

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"normalized": DataType.ARRAY}

    def config_params():
        return {"welford": {"biased_std": False, "outlier_stds": 4.0, "reset": BoolParam(False, trigger=True)}}

    def setup(self):
        self.mean = {}
        self.m2 = {}

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        val = np.asarray(data.data)  # use asarray instead of squeeze to maintain dimensionality
        if val.ndim > 2:
            print("Error: WelfordsZTransform only accepts 1D or 2D arrays")
            return None

        normalized_value = np.zeros_like(val)

        if self.params["welford"]["reset"].value is True:
            self.reset()
        if val.ndim == 1:
            iterator = enumerate(val)
        else:  # if 2D array, apply the transform along the last dimension
            iterator = np.ndenumerate(val)

        for idx, single_val in iterator:
            key = str(idx)
            if key not in self.mean:
                self.mean[key] = {"mean": 0.0, "count": 0}
                self.m2[key] = 0.0

            count = self.mean[key]["count"]
            biased = self.params["welford"]["biased_std"].value
            outlier_stds = self.params["welford"]["outlier_stds"].value

            delta = single_val - self.mean[key]["mean"]
            count += 1
            new_mean = self.mean[key]["mean"] + delta / count
            delta2 = single_val - new_mean
            self.m2[key] += delta * delta2

            if count < self.INIT_STEPS:
                if val.ndim == 1:
                    normalized_value[idx] = 0
                else:
                    normalized_value[idx[0], idx[1]] = 0
            else:
                std_dev = np.sqrt(self.m2[key] / (count if biased else count - 1) + 1e-8)
                if abs(delta) < outlier_stds * std_dev:
                    if val.ndim == 1:
                        normalized_value[idx] = delta / std_dev
                    else:
                        normalized_value[idx[0], idx[1]] = delta / std_dev
                else:
                    if val.ndim == 1:
                        normalized_value[idx] = outlier_stds * np.sign(delta)
                    else:
                        normalized_value[idx[0], idx[1]] = outlier_stds * np.sign(delta)

            self.mean[key] = {"mean": new_mean, "count": count}

        return {"normalized": (normalized_value, data.meta)}

    def reset(self):
        self.mean = {}
        self.m2 = {}
