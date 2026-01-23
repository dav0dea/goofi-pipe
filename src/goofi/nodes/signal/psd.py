import numpy as np
from numpy.fft import fft, fftfreq

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class PSD(Node):
    """
    This node computes the Power Spectral Density (PSD) of input array data using either the FFT or Welch method. It processes one- or two-dimensional input data and returns the PSD values along with frequency information, within a specified frequency range.

    Inputs:
    - data: An array (1D or 2D) containing the signal data to analyze, with associated metadata including sampling frequency.

    Outputs:
    - psd: An array representing the power spectral density of the input data, along with updated metadata including the selected frequency values.
    """

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
                "nperseg": FloatParam(0.5, 0.0, 10000.0, doc="Segment length. Unit determined by nperseg_unit."),
                "nperseg_unit": StringParam(
                    "fraction",
                    options=["samples", "seconds", "fraction"],
                    doc="Unit for nperseg: 'samples' (data points), 'seconds' (requires sfreq in metadata), or 'fraction' (fraction of signal length).",
                ),
                "noverlap": FloatParam(0.75, 0.0, 10000.0, doc="Overlap length. Unit determined by noverlap_unit."),
                "noverlap_unit": StringParam(
                    "fraction_nperseg",
                    options=["samples", "seconds", "fraction", "fraction_nperseg"],
                    doc="Unit for noverlap: 'samples' (data points), 'seconds' (requires sfreq in metadata), 'fraction' (fraction of signal length), or 'fraction_nperseg' (fraction of nperseg).",
                ),
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
        axis = (
            self.params.psd.axis.value
            if self.params.psd.axis.value >= 0
            else data.data.ndim + self.params.psd.axis.value
        )

        sfreq = data.meta["sfreq"]
        signal_length = data.data.shape[axis]

        # Calculate nperseg based on unit
        nperseg_value = self.params.welch.nperseg.value
        nperseg_unit = self.params.welch.nperseg_unit.value

        if nperseg_unit == "samples":
            nperseg = int(nperseg_value) if nperseg_value > 0 else None
        elif nperseg_unit == "seconds":
            nperseg = int(nperseg_value * sfreq) if nperseg_value > 0 else None
        elif nperseg_unit == "fraction":
            nperseg = int(nperseg_value * signal_length) if nperseg_value > 0 else None
        else:
            raise ValueError(f"Unknown nperseg_unit: {nperseg_unit}")

        # Calculate noverlap based on unit
        noverlap_value = self.params.welch.noverlap.value
        noverlap_unit = self.params.welch.noverlap_unit.value

        if noverlap_unit == "samples":
            noverlap = int(noverlap_value) if noverlap_value >= 0 else None
        elif noverlap_unit == "seconds":
            noverlap = int(noverlap_value * sfreq) if noverlap_value >= 0 else None
        elif noverlap_unit == "fraction":
            # Fraction is of signal length
            noverlap = int(noverlap_value * signal_length) if noverlap_value >= 0 else None
        elif noverlap_unit == "fraction_nperseg":
            # Fraction is of nperseg
            if nperseg is not None:
                noverlap = int(noverlap_value * nperseg) if noverlap_value >= 0 else None
            else:
                noverlap = None
        else:
            raise ValueError(f"Unknown noverlap_unit: {noverlap_unit}")

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
