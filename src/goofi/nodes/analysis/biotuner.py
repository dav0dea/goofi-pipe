import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class Biotuner(Node):
    """
    This node extracts dominant frequency peaks from an incoming 1D signal and computes several music-theoretical and harmonic metrics from the spectral content using the Biotuner library. It provides both raw peak data and a variety of measures relating to harmonicity and consonance, potentially useful for music analysis, neurofeedback, or real-time audio investigations.

    Inputs:
    - data: A 1D array containing the signal from which spectral peaks and associated metrics will be derived.

    Outputs:
    - harmsim: Harmonic similarity metric for the extracted peaks.
    - tenney: Tenney height metric, a consonance measure, for the peaks.
    - subharm_tension: Subharmonic tension metric, indicating harmonic tension based on subharmonics.
    - cons: General consonance value computed from the peaks.
    - peaks_ratios_tuning: Ratios between the detected peaks, reflecting interval relationships for tuning.
    - harm_tuning: Harmonic tuning information for the peaks.
    - peaks: Array of main frequency peaks detected in the input signal.
    - amps: Amplitudes corresponding to the detected main frequency peaks.
    - extended_peaks: Additional peaks derived by harmonic extension methods.
    - extended_amps: Amplitudes of the extended peaks.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "harmsim": DataType.ARRAY,
            "tenney": DataType.ARRAY,
            "subharm_tension": DataType.ARRAY,
            "cons": DataType.ARRAY,
            "peaks_ratios_tuning": DataType.ARRAY,
            "harm_tuning": DataType.ARRAY,
            "peaks": DataType.ARRAY,
            "amps": DataType.ARRAY,
            "extended_peaks": DataType.ARRAY,
            "extended_amps": DataType.ARRAY,
        }

    def config_params():
        return {
            "biotuner": {
                "n_peaks": IntParam(5, 1, 10, doc="Number of peaks to extract"),
                "f_min": FloatParam(2.0, 0.1, 50.0, doc="Minimum frequency"),
                "f_max": FloatParam(30.0, 1.0, 100.0, doc="Maximum frequency"),
                "precision": FloatParam(0.1, 0.01, 10.0, doc="Precision of the peak extraction in Hz"),
                "peaks_function": StringParam(
                    "EMD", options=["EMD", "fixed", "harmonic_recurrence", "EIMC"], doc="Peak extraction function"
                ),
                "n_harm_subharm": IntParam(3, 1, 10, doc="Number of harmonics to consider in the subharmonic tension metric"),
                "n_harm_extended": IntParam(3, 1, 10, doc="Number of harmonics to consider in the extended peaks"),
                "delta_lim": IntParam(250, 1, 300, doc="Maximum delta (in ms) in the subharmonic tension metric"),
            }
        }

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim > 1:
            raise ValueError("Data must be 1D")

        result = biotuner_realtime(
            data.data,
            data.meta["sfreq"],
            n_peaks=self.params["biotuner"]["n_peaks"].value,
            peaks_function=self.params["biotuner"]["peaks_function"].value,
            min_freq=self.params["biotuner"]["f_min"].value,
            max_freq=self.params["biotuner"]["f_max"].value,
            precision=self.params["biotuner"]["precision"].value,
            n_harm_extended=self.params["biotuner"]["n_harm_extended"].value,
            n_harm_subharm=self.params["biotuner"]["n_harm_subharm"].value,
            delta_lim=self.params["biotuner"]["delta_lim"].value,
        )
        peaks, extended_peaks, metrics, tuning, harm_tuning, amps, extended_amps = result

        return {
            "harmsim": (np.array([metrics["harmsim"]]), data.meta),
            "tenney": (np.array([metrics["tenney"]]), data.meta),
            "subharm_tension": (np.array([metrics["subharm_tension"][0]]), data.meta),
            "cons": (np.array([metrics["cons"]]), data.meta),
            "peaks_ratios_tuning": (np.array(tuning), data.meta),
            "harm_tuning": (np.array(harm_tuning), data.meta),
            "peaks": (np.array(peaks), data.meta),
            "amps": (np.array(amps), data.meta),
            "extended_peaks": (np.array(extended_peaks), data.meta),
            "extended_amps": (np.array(extended_amps), data.meta),
        }


compute_biotuner_fn, harmonic_tuning_fn = None, None


def biotuner_realtime(
    data,
    sfreq,
    n_peaks=5,
    peaks_function="EMD",
    min_freq=1,
    max_freq=65,
    precision=0.1,
    n_harm_extended=3,
    n_harm_subharm=3,
    delta_lim=250,
):
    # import the biotuner function here to avoid loading it on startup
    global compute_biotuner_fn, harmonic_tuning_fn
    if compute_biotuner_fn is None or harmonic_tuning_fn is None:
        from biotuner.biotuner_object import compute_biotuner, harmonic_tuning

        compute_biotuner_fn = compute_biotuner
        harmonic_tuning_fn = harmonic_tuning

    # run biotuner peak extraction
    bt = compute_biotuner_fn(peaks_function=peaks_function, sf=sfreq)
    try:
        bt.peaks_extraction(
            np.array(data),
            graph=False,
            min_freq=min_freq,
            max_freq=max_freq,
            precision=precision,
            nIMFs=5,
            n_peaks=n_peaks,
            smooth_fft=1,
        )
    except UnboundLocalError:
        raise RuntimeError("No peaks found. Try increasing the length of the signal.")

    try:
        # try computing the extended peaks
        bt.peaks_extension(method="harmonic_fit", n_harm=n_harm_extended)
    except TypeError:
        raise RuntimeError("Detected only one peak. The largest peak might be below the minimum frequency.")

    bt.compute_peaks_metrics(n_harm=n_harm_subharm, delta_lim=delta_lim)
    if hasattr(bt, "all_harmonics"):
        harm_tuning = harmonic_tuning_fn(bt.all_harmonics)
    else:
        harm_tuning = [0, 1]
    peaks = bt.peaks
    amps = bt.amps
    extended_peaks = bt.extended_peaks
    extended_amps = bt.extended_amps
    metrics = bt.peaks_metrics

    if not isinstance(metrics["subharm_tension"][0], float):
        metrics["subharm_tension"] = [np.nan]
    metrics["harmsim"] = metrics["harmsim"] / 100
    # rescale tenney height from between 4 to 9 to between 0 and 1
    metrics["tenney"] = (metrics["tenney"] - 4) / 5
    tuning = bt.peaks_ratios
    return peaks, extended_peaks, metrics, tuning, harm_tuning, amps, extended_amps
