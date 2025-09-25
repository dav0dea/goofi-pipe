import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class LempelZiv(Node):
    """
    This node computes the Lempel-Ziv complexity (LZC) of input array data. The LZC is a measure of complexity in a binary sequence, often used in signal analysis to quantify regularity or randomness. The input array is binarized using either a mean or median threshold along a specified axis, and then the Lempel-Ziv complexity is computed for each segment. The output is an array of LZC values, retaining the input metadata.

    Inputs:
    - data: Input array data to be analyzed.

    Outputs:
    - lzc: Array containing the Lempel-Ziv complexity values calculated from the input data, with metadata attached.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"lzc": DataType.ARRAY}

    def config_params():
        return {
            "lempel_ziv": {
                "binarization": StringParam("mean", options=["mean", "median"]),
                "axis": -1,
            }
        }

    def setup(self):
        from antropy import lziv_complexity

        self.compute_lzc = lziv_complexity

    def process(self, data: Data):
        if data is None:
            # no data, skip processing
            return None

        # read parameters
        binarize_mode = self.params.lempel_ziv.binarization.value
        axis = self.params.lempel_ziv.axis.value

        # binarize data
        if binarize_mode == "mean":
            binarized = data.data > np.mean(data.data, axis=axis, keepdims=True)  # mean split
        elif binarize_mode == "median":
            binarized = data.data > np.median(data.data, axis=axis, keepdims=True)  # median split
        else:
            raise ValueError(f"Unknown binarize mode: {binarize_mode}")

        # compute Lempel-Ziv complexity
        lzc = np.apply_along_axis(self.compute_lzc, axis, binarized, normalize=True)

        axis = axis if axis >= 0 else axis + data.data.ndim
        if "channels" in data.meta:
            if f"dim{axis}" in data.meta["channels"]:
                del data.meta["channels"][f"dim{axis}"]
            for i in range(axis + 1, data.data.ndim):
                if f"dim{i}" in data.meta["channels"]:
                    data.meta["channels"][f"dim{i-1}"] = data.meta["channels"].pop(f"dim{i}")

        # return Lempel-Ziv complexity and incoming metadata
        return {"lzc": (lzc, data.meta)}
