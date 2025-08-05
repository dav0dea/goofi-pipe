import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam


class ResampleJoint(Node):
    """
    This node jointly resamples two input arrays so that both are converted to the same new sampling frequency, computed as an interpolation between their original sampling rates. The original data arrays are resampled to this common frequency and outputted along with updated metadata.

    Inputs:
    - data1: First input array along with its associated metadata, including original sampling frequency ("sfreq").
    - data2: Second input array along with its associated metadata, including original sampling frequency ("sfreq").

    Outputs:
    - out1: Resampled version of data1, matched to the new common sampling frequency with updated metadata.
    - out2: Resampled version of data2, matched to the new common sampling frequency with updated metadata.
    """

    def config_input_slots():
        # Defining two input slots for two input signals
        return {"data1": DataType.ARRAY, "data2": DataType.ARRAY}

    def config_output_slots():
        # Defining two output slots for the resampled signals
        return {"out1": DataType.ARRAY, "out2": DataType.ARRAY}

    def config_params():
        return {"resample": {"scale": FloatParam(0.5, 0.0, 1.0)}}

    def setup(self):
        from scipy.signal import resample_poly

        self.resample_poly = resample_poly

    def process(self, data1: Data, data2: Data):
        if data1 is None or data1.data is None or data2 is None or data2.data is None:
            return None

        sf1 = data1.meta["sfreq"]
        sf2 = data2.meta["sfreq"]

        # Find the new sampling frequency based on scale
        scale = self.params.resample.scale.value
        new_sfreq = scale * min(sf1, sf2) + (1 - scale) * max(sf1, sf2)

        # Calculate the resampling factors
        up1, down1 = int(new_sfreq), int(sf1)
        up2, down2 = int(new_sfreq), int(sf2)

        signal1 = np.array(data1.data)
        signal2 = np.array(data2.data)

        # Resample the signals
        resampled_signal1 = self.resample_poly(signal1, up1, down1, padtype="line")
        resampled_signal2 = self.resample_poly(signal2, up2, down2, padtype="line")

        # Update the 'sfreq' metadata
        data1.meta["sfreq"] = new_sfreq
        data2.meta["sfreq"] = new_sfreq

        return {"out1": (resampled_signal1, data1.meta), "out2": (resampled_signal2, data2.meta)}
