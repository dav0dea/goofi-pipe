from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam


class EMD(Node):
    """
    Applies Empirical Mode Decomposition (EMD) to a one-dimensional array input signal, extracting its intrinsic mode functions (IMFs). The node returns the resulting IMFs as a multi-channel array, with each channel corresponding to an individual IMF. IMF indices are added to the metadata for channel identification.

    Inputs:
    - data: A one-dimensional array representing the input signal to be decomposed.

    Outputs:
    - IMFs: An array containing the extracted intrinsic mode functions, with channel metadata indicating the IMF index.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"IMFs": DataType.ARRAY}

    def config_params():
        return {"emd": {"nIMFs": IntParam(5, 1, 10)}}

    def setup(self):
        from biotuner.peaks_extraction import EMD_eeg

        self.EMD_eeg = EMD_eeg

    def process(self, data: Data):
        if data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        n_imfs = self.params.emd.nIMFs.value

        # add indices for each IMF in the meta data as strings
        imfs = self.EMD_eeg(data.data, method="EMD_fast", graph=False, extrema_detection="simple", nIMFs=5)[:n_imfs]
        data.meta["channels"]["dim0"] = ["IMF" + str(i) for i in range(imfs.shape[0])]

        return {"IMFs": (imfs, data.meta)}
