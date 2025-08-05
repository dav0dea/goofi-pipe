import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Reduce(Node):
    """
    Reduces an input array along a specified axis using a selected reduction operation. The node supports various reduction methods such as mean, median, min, max, standard deviation, norm, and sum. After reduction, it also updates the metadata to reflect changes in dimensionality.

    Inputs:
    - array: Multidimensional array data to be reduced.

    Outputs:
    - out: The array after reduction and its updated metadata.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "reduce": {
                "method": StringParam(
                    "mean",
                    options=["mean", "median", "min", "max", "std", "norm", "sum"],
                ),
                "axis": 0,
            }
        }

    def process(self, array: Data):
        if array is None:
            return None

        axis = self.params.reduce.axis.value
        if axis < 0:
            axis = array.data.ndim + axis

        if self.params.reduce.method.value == "norm":
            result = np.linalg.norm(array.data, axis=axis)
        else:
            result = getattr(np, self.params.reduce.method.value)(array.data, axis=axis)

        if f"dim{axis}" in array.meta["channels"]:
            del array.meta["channels"][f"dim{axis}"]

        for i in range(axis, result.ndim + 1):
            if f"dim{i+1}" in array.meta["channels"]:
                array.meta["channels"][f"dim{i}"] = array.meta["channels"][f"dim{i+1}"]
                del array.meta["channels"][f"dim{i+1}"]

        return {"out": (result, array.meta)}
