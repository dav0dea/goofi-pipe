import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam


class DissonanceCurve(Node):
    """
    This node calculates a dissonance curve for a set of spectral peaks and their amplitudes, providing a quantitative measure of sensory dissonance as a function of interval ratios. It is typically used to analyze the dissonance of complex tones and suggest tunings that minimize dissonance.

    Inputs:
    - peaks: An array of frequency peaks (typically from a spectrum).
    - amps: An array of amplitudes corresponding to the frequency peaks.

    Outputs:
    - dissonance_curve: An array representing the computed dissonance curve over a range of interval ratios.
    - tuning: An array of interval ratios (relative tunings) corresponding to the dissonance curve.
    - avg_dissonance: An array containing the average dissonance value across tested intervals.
    """

    def config_input_slots():
        return {
            "peaks": DataType.ARRAY,
            "amps": DataType.ARRAY,
        }

    def config_output_slots():
        return {"dissonance_curve": DataType.ARRAY, "tuning": DataType.ARRAY, "avg_dissonance": DataType.ARRAY}

    def config_params():
        return {
            "Dissonance_Curve": {
                "max_ratio": FloatParam(
                    2.0, 1.0, 10.0, doc="Maximum ratio to extend the dissonance curve to; 2 represents the octave"
                ),
                "max_denom": IntParam(100, 10, 1000, doc="Maximum denominator for rational numbers in the dissonance curve"),
            }
        }

    def setup(self):
        from biotuner.scale_construction import diss_curve

        self.diss_curve = diss_curve

    def process(self, peaks: Data, amps: Data):
        if peaks is None or peaks.data is None or amps is None or amps.data is None:
            return None
        peaks.data = np.squeeze(peaks.data)
        metadata = peaks.meta
        peaks = peaks.data
        amps = amps.data
        denom = self.params["Dissonance_Curve"]["max_denom"].value
        max_ratio = self.params["Dissonance_Curve"]["max_ratio"].value
        if peaks.data.ndim == 1:
            peaks = [p * 128 for p in peaks]  # scale the peaks up to accomodate beating frequency modelling.
            amps = np.interp(amps, (np.array(amps).min(), np.array(amps).max()), (0.2, 0.8))

            diss, intervals, diss_scale, euler_diss, diss_mean, harm_sim_diss = self.diss_curve(
                peaks,
                amps,
                denom=denom,
                max_ratio=max_ratio,
                euler_comp=False,
                method="min",
                plot=False,
                n_tet_grid=12,
            )
            # rescale diss to 0-1
            diss = np.interp(diss, (np.array(diss).min(), np.array(diss).max()), (0, 1))
            return {
                "dissonance_curve": (np.array(diss), metadata),
                "tuning": (np.array(diss_scale), metadata),
                "avg_dissonance": (np.array(diss_mean), metadata),
            }
        elif peaks.data.ndim == 2:
            raise NotImplementedError("Dissonance curve for multiple signals not implemented yet.")
