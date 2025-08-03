import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, BoolParam


class Math(Node):
    """
    This node performs a series of mathematical operations and rescaling on array data. It processes incoming arrays by applying configurable arithmetic operations, rounding, and optional power functions, and then maps the result from a specified input range to an output range. This allows flexible numerical manipulation and transformation of real-time signals.

    Inputs:
    - data: An array containing the signal or values to be processed.

    Outputs:
    - out: The processed array after all mathematical operations and rescaling, along with its metadata.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "math": {
                "pre_add": FloatParam(0.0, -10.0, 10.0),
                "multiply": FloatParam(1.0, -10.0, 10.0),
                "post_add": FloatParam(0.0, -10.0, 10.0),
                "round": IntParam(-1, -1, 10),
                "sqrt": BoolParam(False),
                "squared": BoolParam(False),
            },
            "map": {
                "input_min": FloatParam(0.0, -10.0, 10.0),
                "input_max": FloatParam(1.0, -10.0, 10.0),
                "output_min": FloatParam(0.0, -10.0, 10.0),
                "output_max": FloatParam(1.0, -10.0, 10.0),
            },
        }

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        signal = data.data

        # apply math operations
        signal += self.params["math"]["pre_add"].value
        signal *= self.params["math"]["multiply"].value
        signal += self.params["math"]["post_add"].value

        # apply rounding
        decimals = self.params["math"]["round"].value
        if decimals >= 0:
            signal = np.around(signal, decimals)

        # rescale signal from input range to output range
        signal = self.rescale(
            signal,
            self.params["map"]["input_min"].value,
            self.params["map"]["input_max"].value,
            self.params["map"]["output_min"].value,
            self.params["map"]["output_max"].value,
        )
        if self.params["math"]["sqrt"].value:
            signal = np.sqrt(signal)
        if self.params["math"]["squared"].value:
            signal = np.square(signal)
        return {"out": (signal, data.meta)}

    @staticmethod
    def rescale(signal, input_min, input_max, output_min, output_max):
        # rescale signal from input range to output range
        return ((signal - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
