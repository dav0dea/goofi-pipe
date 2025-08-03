from math import gcd

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam


class Resample(Node):
    """
    This node resamples an input signal array from its original sampling frequency to a new specified sampling frequency using polyphase filtering. It supports resampling along any axis of the input array. The node ensures any invalid numerical values (such as NaN or infinity) in the input are replaced with zeros before processing. The output array has its sample rate updated and, if applicable, the relevant channel metadata for the resampled axis is removed.

    Inputs:
    - data: Array data to be resampled, along with its associated metadata including sampling frequency and channel information.

    Outputs:
    - out: The resampled array with updated metadata reflecting the new sampling frequency and any changes to channel information.
    """

    def config_input_slots():
        # Defining one input slot for one input signal
        return {"data": DataType.ARRAY}

    def config_output_slots():
        # Defining one output slot for the resampled signal
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "resample": {
                "new_sfreq": IntParam(1000, 10, 44100),  # New sampling frequency as a parameter
                "axis": IntParam(-1, -1, 4),  # Axis to resample the signal
            }
        }

    def setup(self):
        from scipy.signal import resample_poly

        self.resample_poly = resample_poly

    def process(self, data: Data):
        if data is None or data.data is None:
            print("Data is None")
            return None

        # Original sampling frequency from metadata
        sf = data.meta["sfreq"]

        # Retrieve new sampling frequency parameter
        new_sfreq = self.params["resample"]["new_sfreq"].value
        axis = self.params["resample"]["axis"].value

        # Calculate up and down factors based on the gcd of sf and new_sfreq
        factor = gcd(int(sf), int(new_sfreq))
        up = new_sfreq // factor
        down = sf // factor

        signal = np.array(data.data)

        # replace NaNs and Infs with zeros
        signal = np.nan_to_num(signal, posinf=0, neginf=0)

        # Resample the signal based on the specified axis
        resampled_signal = self.resample_poly(signal, up, down, axis=axis, padtype="line")

        # Adjust axis for negative values
        if axis < 0:
            axis += signal.ndim

        # Remove the corresponding channel metadata if it exists
        if f"dim{axis}" in data.meta["channels"]:
            del data.meta["channels"][f"dim{axis}"]

        # Update the 'sfreq' metadata
        data.meta["sfreq"] = new_sfreq

        return {"out": (resampled_signal, data.meta)}
