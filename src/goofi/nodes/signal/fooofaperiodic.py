import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class FOOOFaperiodic(Node):
    """
    This node extracts the aperiodic parameters and peak center frequencies from input power spectral density (PSD) data using the FOOOF algorithm. It fits FOOOF models to each spectrum, separates the aperiodic (background) component from periodic peaks, and outputs both the background parameters and the PSD with the peaks removed.

    Inputs:
    - psd_data: Input array containing one or more power spectra with associated frequency information.

    Outputs:
    - offset: The offset parameter of the aperiodic (background) component for each spectrum.
    - exponent: The exponent parameter of the aperiodic (background) component for each spectrum.
    - cf_peaks: The center frequencies of all detected peaks in each spectrum.
    - cleaned_psd: The power spectrum with periodic peaks removed, containing the aperiodic component only.
    """

    def config_input_slots():
        return {"psd_data": DataType.ARRAY}

    def config_output_slots():
        return {
            "offset": DataType.ARRAY,
            "exponent": DataType.ARRAY,
            "cf_peaks": DataType.ARRAY,
            "cleaned_psd": DataType.ARRAY,
        }

    def config_params():
        return {
            "fooof": {
                "max_n_peaks": IntParam(-1, 1, 20, doc="The maximum number of peaks to fit."),
                "mode": StringParam("fixed", options=["fixed", "knee"], doc="The mode to fit the aperiodic component."),
                "freq_min": FloatParam(-1.0, 0.0, 512.0, doc="The minimum frequency to consider for fitting."),
                "freq_max": FloatParam(-1.0, 0.0, 512.0, doc="The maximum frequency to consider for fitting."),
                "peak_width_min": FloatParam(0.5, 0.0, 100.0, doc="The minimum width of a peak, in Hz."),
                "peak_width_max": FloatParam(12.0, 0.0, 512.0, doc="The maximum width of a peak, in Hz."),
            }
        }

    def setup(self):
        from fooof import FOOOF

        self.FOOOF = FOOOF

    def process(self, psd_data: Data):
        if psd_data is None or psd_data.data is None:
            return None

        ch_annot = None
        try:
            if psd_data.data.ndim == 1:
                freqs = np.array(psd_data.meta["channels"]["dim0"])
                ch_annot = psd_data.meta["channels"]["dim1"] if "dim1" in psd_data.meta["channels"] else None
            elif psd_data.data.ndim == 2:
                freqs = np.array(psd_data.meta["channels"]["dim1"])
                ch_annot = psd_data.meta["channels"]["dim0"] if "dim0" in psd_data.meta["channels"] else None
            else:
                raise ValueError("Invalid data shape. Expected 1D or 2D array.")
        except KeyError:
            raise ValueError("No frequency information. Make sure to pass a power spectrum with frequency information.")

        max_n_peaks = self.params.fooof.max_n_peaks.value if self.params.fooof.max_n_peaks.value > 0 else float("inf")
        freq_range = [
            self.params.fooof.freq_min.value if self.params.fooof.freq_min.value > 0 else freqs.min(),
            self.params.fooof.freq_max.value if self.params.fooof.freq_max.value > 0 else freqs.max(),
        ]

        offsets, exponents, cf_peaks, cleaned_psds = [], [], [], []
        for psd in psd_data.data:
            try:
                # fit FOOOF model
                fm = self.FOOOF(
                    peak_width_limits=(self.params.fooof.peak_width_min.value, self.params.fooof.peak_width_max.value),
                    max_n_peaks=max_n_peaks,
                    aperiodic_mode=self.params.fooof.mode.value,
                )
                fm.fit(freqs, psd, freq_range=freq_range)
            except Exception as e:
                offsets.append(np.nan)
                exponents.append(np.nan)
                cf_peaks.append(np.array([]))
                cleaned_psds.append(np.array([]))
                continue

            # extract aperiodic component and peak parameters
            aperiodic_params = fm.get_params("aperiodic_params")
            peak_params = fm.get_params("peak_params")
            cleaned_psd = fm._spectrum_peak_rm

            # extracting specific parameters
            offset = aperiodic_params[0]
            exponent = aperiodic_params[-1]
            # getting all center frequencies of detected peaks
            cf_peak = peak_params[:, 0] if peak_params.size else np.array([])

            # add to lists
            offsets.append(offset)
            exponents.append(exponent)
            cf_peaks.append(cf_peak)
            cleaned_psds.append(cleaned_psd)

        # convert lists to numpy arrays
        offsets = np.array(offsets)
        exponents = np.array(exponents)
        cf_peaks = pad_to_max_len(cf_peaks)
        cleaned_psds = pad_to_max_len(cleaned_psds)

        return {
            "offset": (offsets, {"channels": {"dim0": ch_annot}, "sfreq": psd_data.meta["sfreq"]}),
            "exponent": (exponents, {"channels": {"dim0": ch_annot}, "sfreq": psd_data.meta["sfreq"]}),
            "cf_peaks": (cf_peaks, {"channels": {"dim0": ch_annot}, "sfreq": psd_data.meta["sfreq"]}),
            "cleaned_psd": (cleaned_psds, {"channels": {"dim0": ch_annot}, "sfreq": psd_data.meta["sfreq"]}),
        }


def pad_to_max_len(arrays):
    max_len = max([len(arr) for arr in arrays], default=0)
    padded_arrays = [np.pad(arr, (0, max_len - len(arr)), "constant", constant_values=np.nan) for arr in arrays]
    return np.array(padded_arrays)
