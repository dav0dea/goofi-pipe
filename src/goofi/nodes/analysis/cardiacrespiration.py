from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class CardiacRespiration(Node):
    """
    This node extracts a respiration signal derived from cardiac activity in a 1D physiological waveform. It processes either ECG or (in the future) PPG signals and returns an estimated respiratory waveform (EDR) based on cardiac information.

    Inputs:
    - data: 1D array containing a physiological waveform (such as ECG), including associated metadata like sampling rate.

    Outputs:
    - cardiac: 1D array containing the cardiac-derived respiration (EDR) signal with original metadata.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"cardiac": DataType.ARRAY}

    def config_params():
        return {
            "cardiac": {
                "input_type": StringParam("ppg", options=["ppg", "ecg"]),
            }
        }

    def setup(self):
        import neurokit2 as nk

        self.neurokit = nk

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        if self.params["cardiac"]["input_type"].value == "ppg":
            raise NotImplementedError("PPG not implemented yet")
            # signal, info = self.neurokit.ppg_process(data.data, sampling_rate=data.meta["sfreq"])
            # hrv_df = self.neurokit.hrv(info, sampling_rate=data.meta["sfreq"])
        elif self.params["cardiac"]["input_type"].value == "ecg":
            # extract peaks
            rpeaks, info = self.neurokit.ecg_peaks(data.data, sampling_rate=data.meta["sfreq"])
            # compute rate
            ecg_rate = self.neurokit.ecg_rate(rpeaks, sampling_rate=data.meta["sfreq"], desired_length=len(data.data))
            edr = self.neurokit.ecg_rsp(ecg_rate, sampling_rate=data.meta["sfreq"])

        return {"cardiac": (edr, data.meta)}
