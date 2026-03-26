import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node


class Transpose(Node):

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"transpose": {"axes": "1,0"}}

    def process(self, array: Data):
        if array is None or array.data is None:
            return None

        if array.data.ndim == 1:
            # If the input is a 1D array, add an extra dimension
            result = array.data.reshape(1, -1)
        else:
            axes = list(map(lambda a: int(a), self.params.transpose.axes.value.split(",")))
            result = np.transpose(array.data, axes=axes)

        # transpose channel names
        ch_names = {}
        if "dim0" in array.meta["channels"]:
            ch_names["dim1"] = array.meta["channels"]["dim0"]
        if "dim1" in array.meta["channels"]:
            ch_names["dim0"] = array.meta["channels"]["dim1"]
        array.meta["channels"] = ch_names

        return {"out": (result, {})}  # TODO: fix meta axes
