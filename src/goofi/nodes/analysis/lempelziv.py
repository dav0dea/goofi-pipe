import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class LempelZiv(Node):
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

        # compute Lempel-Ziv complexity
        lzc = np.apply_along_axis(self.compute_lzc, axis, binarized, normalize=True)

        # return Lempel-Ziv complexity and incoming metadata
        return {"lzc": (lzc, data.meta)}
