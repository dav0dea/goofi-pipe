import numpy as np

from goofi.data import DataType
from goofi.node import Node
from goofi.params import FloatParam, StringParam


# TODO: deal meta data output
class PowerBand(Node):
    """
    This node calculates the total power within a specified frequency band from input data representing a power spectral density (PSD). It sums the power values within a chosen frequency range to output either the absolute or relative band power. The output preserves the input's metadata.

    Inputs:
    - data: An array containing PSD values with associated frequency information in the metadata.

    Outputs:
    - power: The computed band power as an array, along with the original metadata.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"power": DataType.ARRAY}

    def config_params():
        return {
            "powerband": {
                "f_min": FloatParam(1.0, 0.01, 9999.0),
                "f_max": FloatParam(60.0, 1.0, 10000.0),
                "power_type": StringParam("absolute", options=["absolute", "relative"]),
            }
        }

    def process(self, data):
        if data is None or data.data is None:
            return None

        f_min = self.params["powerband"]["f_min"].value
        f_max = self.params["powerband"]["f_max"].value
        power_type = self.params["powerband"]["power_type"].value
        if "dim1" in data.meta["channels"]:
            freq = np.array(data.meta["channels"]["dim1"])
        else:
            freq = np.array(data.meta["channels"]["dim0"])

        # Selecting the range of frequencies
        valid_indices = np.where((freq >= f_min) & (freq <= f_max))[0]
        if data.data.ndim == 1:
            selected_psd = data.data[valid_indices]
        else:  # if 2D
            selected_psd = data.data[:, valid_indices]

        # Computing the power
        power = np.sum(selected_psd, axis=-1)
        if power_type == "relative":
            total_power = np.sum(data.data, axis=-1)
            power = power / total_power

        return {"power": (np.array(power), data.meta)}
