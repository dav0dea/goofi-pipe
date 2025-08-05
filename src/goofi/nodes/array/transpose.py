from goofi.data import Data, DataType
from goofi.node import Node


class Transpose(Node):
    """
    Transposes a 2D array input, swapping its rows and columns. If the input is a 1D array, it is first converted to a 2D column vector before transposing. The node also swaps the associated "dim0" and "dim1" channel metadata to reflect the transpose operation.

    Inputs:
    - array: An array input, expected to be 1D or 2D. If 1D, it is reshaped to 2D before transposing.

    Outputs:
    - out: The transposed array along with updated channel metadata reflecting the swapped dimensions.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {}

    def process(self, array: Data):
        if array is None or array.data is None:
            return None

        if array.data.ndim == 1:
            # If the input is a 1D array, add an extra dimension
            array.data = array.data.reshape(-1, 1)

        if array.data.ndim != 2:
            # TODO: support n-dimensional arrays
            raise ValueError("Data must be 2D (TODO: support n-dimensional arrays).")

        result = array.data.T

        # transpose channel names
        ch_names = {}
        if "dim0" in array.meta["channels"]:
            ch_names["dim1"] = array.meta["channels"]["dim0"]
        if "dim1" in array.meta["channels"]:
            ch_names["dim0"] = array.meta["channels"]["dim1"]
        array.meta["channels"] = ch_names

        return {"out": (result, array.meta)}
