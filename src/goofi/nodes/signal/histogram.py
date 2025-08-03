import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class Histogram(Node):
    """
    This node computes the histogram of an input array using either traditional binning or kernel density estimation (KDE). It receives an array of data, flattens it, computes a histogram or KDE within a specified data range, and outputs the resulting distribution along with metadata describing each bin.

    Inputs:
    - data: An array of numerical values to be analyzed.

    Outputs:
    - histogram: The computed histogram or KDE values, along with metadata including the bin lower edges as channel names.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"histogram": DataType.ARRAY}

    def config_params():
        return {
            "histogram": {
                "method": StringParam("bins", options=["bins", "kde"]),
                "bin_count": IntParam(100, 2, 200),
                "data_range": StringParam("auto", options=["auto", "manual", "minmax"]),
                "min_val": FloatParam(-1.0, -10.0, 10.0),
                "max_val": FloatParam(1.0, -10.0, 10.0),
                "reset_data_range": BoolParam(False, trigger=True),
            }
        }

    def setup(self):
        self.data_min = None
        self.data_max = None

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        # Handle reset trigger
        if self.params.histogram.reset_data_range.value:
            self.data_min = None
            self.data_max = None

        # Flatten data for histogram computation
        data_flat = data.data.flatten()

        # Determine data range
        data_range = self.params.histogram.data_range.value
        if data_range == "manual":
            range_min = self.params.histogram.min_val.value
            range_max = self.params.histogram.max_val.value
        elif data_range == "minmax":
            range_min = np.min(data_flat)
            range_max = np.max(data_flat)
        else:  # "auto"
            if self.data_min is None or self.data_max is None:
                self.data_min = np.min(data_flat)
                self.data_max = np.max(data_flat)
            else:
                self.data_min = min(self.data_min, np.min(data_flat))
                self.data_max = max(self.data_max, np.max(data_flat))
            range_min = self.data_min
            range_max = self.data_max

        # Compute histogram or KDE
        method = self.params.histogram.method.value
        bin_count = self.params.histogram.bin_count.value

        if method == "bins":
            hist, bin_edges = np.histogram(data_flat, bins=bin_count, range=(range_min, range_max))
        else:  # "kde"
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data_flat)
            bin_edges = np.linspace(range_min, range_max, bin_count + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist = kde(bin_centers)

        # Create metadata with bin lower edges as channel names
        meta = data.meta.copy()
        bin_lower_edges = bin_edges[:-1]  # Lower edge of each bin
        meta["channels"] = {"dim0": [f"{edge:.3f}" for edge in bin_lower_edges]}

        return {"histogram": (hist, meta)}
