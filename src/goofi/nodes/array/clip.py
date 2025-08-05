import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class Clip(Node):
    """
    Clips the values of an input array so that they stay within a specified minimum and maximum range. Any values below the minimum are set to the minimum, and any values above the maximum are set to the maximum.

    Inputs:
    - array: The input array containing numerical values to be clipped.

    Outputs:
    - out: The clipped array with all values constrained within the defined range.
    """

    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {"clip": {"min": FloatParam(-1), "max": FloatParam(1)}}

    def process(self, array: Data):
        if array is None:
            return None

        result = np.clip(array.data, self.params.clip.min.value, self.params.clip.max.value)

        return {"out": (result, array.meta)}
