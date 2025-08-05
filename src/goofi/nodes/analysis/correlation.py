import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Correlation(Node):
    """
    Calculates the Pearson correlation coefficient and p-value between two input arrays. This node measures the linear correlation between the inputs and returns both the correlation values and their statistical significance.

    Inputs:
    - data1: First array of data to correlate.
    - data2: Second array of data to correlate, broadcasted to the same shape as data1 if necessary.

    Outputs:
    - pearson: Array of Pearson correlation coefficients between data1 and data2.
    - pval: Array of p-values indicating the statistical significance of the correlation.
    """

    def config_input_slots():
        return {"data1": DataType.ARRAY, "data2": DataType.ARRAY}

    def config_output_slots():
        return {"pearson": DataType.ARRAY, "pval": DataType.ARRAY}

    def config_params():
        return {"correlation": {"axis": -1}}

    def setup(self):
        from scipy.stats import pearsonr

        self.pearsonr = pearsonr

    def process(self, data1: Data, data2: Data):
        if data1 is None or data2 is None:
            return None

        meta = data1.meta

        # broadcast data to same shape
        data1, data2 = np.broadcast_arrays(data1.data, data2.data)
        data = np.stack([data1, data2], axis=0)

        if data.ndim > 3:
            raise ValueError("Correlation only works for 1D and 2D data")

        # update axis param
        axis = self.params.correlation.axis.value
        if axis >= 0:
            axis += 1

        # calculate correlation along axis
        r, p = self.pearsonr(data[0], data[1], axis=axis)

        return {"pearson": (r, meta), "pval": (p, meta)}
