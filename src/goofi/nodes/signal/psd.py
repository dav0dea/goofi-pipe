import numpy as np
from numpy.fft import fft, fftfreq

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class PSD(Node):
    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {"psd": DataType.ARRAY}

    def config_params():
        return {
            "psd": {
                "method": StringParam("welch", options=["fft", "welch"]),
                "f_min": FloatParam(1, 0.0, 9999.0),
                "f_max": FloatParam(40, 1.0, 10000.0),
                "axis": -1,
            },
            "welch": {
                "nperseg": IntParam(-1, 1, 1000),
                "noverlap": IntParam(-1, 0, 1000),
            },
        }

    def setup(self):
        from scipy.signal import welch

        self.welch = welch

    def process(self, data: Data):
        if data is None or data.data is None:
            return None

        if data.data.ndim not in [1, 2]:
            raise ValueError("Data must be 1D or 2D")

        method = self.params.psd.method.value
        f_min = self.params.psd.f_min.value
        f_max = self.params.psd.f_max.value
        axis = self.params.psd.axis.value if self.params.psd.axis.value >= 0 else data.data.ndim + self.params.psd.axis.value

        nperseg = self.params.welch.nperseg.value if self.params.welch.nperseg.value > 0 else None
        noverlap = self.params.welch.noverlap.value if self.params.welch.noverlap.value >= 0 else None

        sfreq = data.meta["sfreq"]

        if method == "fft":
            freq = fftfreq(data.data.shape[axis], 1 / sfreq)
            psd = np.abs(fft(data.data, axis=axis))
        elif method == "welch":
            freq, psd = self.welch(data.data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=axis)

        # selecting the range of frequencies
        if f_min < 0:
            f_min = freq.min()
        if f_max < 0:
            f_max = freq.max()
        valid_indices = np.where((freq >= f_min) & (freq <= f_max))[0]

        meta = data.meta.copy()
        freq = freq[valid_indices]
        psd = np.take(psd, valid_indices, axis=axis)
        meta["channels"][f"dim{axis}"] = freq.tolist()

        return {"psd": (psd, meta)}
