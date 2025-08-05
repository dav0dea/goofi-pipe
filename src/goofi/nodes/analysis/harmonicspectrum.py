from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import IntParam, StringParam


class HarmonicSpectrum(Node):
    """
    This node computes a harmonic spectrum of an input power spectral density (PSD) array by analyzing the harmonic relationships between frequency components. It calculates harmonicity metrics that quantify how harmonically related different spectral components are, outputting both the complete harmonic spectrum as well as summary statistics indicating the maximum and average harmonicity across frequencies.

    Inputs:
    - psd: An array representing the power spectral density of the signal, with frequency information included in its metadata.

    Outputs:
    - harmonic_spectrum: An array representing the harmonicity value at each frequency bin, indicating the degree to which each frequency is harmonically related to the others in the spectrum.
    - max_harmonicity: The maximum harmonicity value found across all frequency bins, summarizing the most harmonically related component in the spectrum.
    - avg_harmonicity: The average harmonicity value across all frequency bins, summarizing the overall harmonicity of the spectrum.
    """

    def config_input_slots():
        return {"psd": DataType.ARRAY}

    def config_output_slots():
        return {"harmonic_spectrum": DataType.ARRAY, "max_harmonicity": DataType.ARRAY, "avg_harmonicity": DataType.ARRAY}

    def config_params():
        return {
            "harmonic": {
                "metric": StringParam("harmsim", options=["harmsim", "subharm_tension"]),
                "normalize": True,
                "power_law_remove": True,
                "n_harms": IntParam(3, 1, 10, doc="Number of harmonics to consider in the subharmonic tension metric"),
                "delta_lim": IntParam(250, 1, 300, doc="Maximum delta in the subharmonic tension metric"),
                "min_notes": IntParam(2, 1, 10, doc="Minimum number of notes in the subharmonic tension metric"),
            }
        }

    def setup(self):
        from biotuner.biotuner_utils import apply_power_law_remove
        from biotuner.metrics import compute_subharmonic_tension, dyad_similarity

        self.apply_power_law_remove = apply_power_law_remove
        self.compute_subharmonic_tension = compute_subharmonic_tension
        self.dyad_similarity = dyad_similarity

        self.cached_freqs = None
        self.cached_harmonicity_matrix = None
        self.cached_metric = None

    def process(self, psd: Data):
        if psd is None or psd.data is None:
            return None
        psd.data = np.squeeze(psd.data)
        metadata = psd.meta

        if psd.data.ndim == 1:
            freqs = metadata["channels"]["dim0"]
            if freqs[0] == 0:
                freqs[0] = 1e-8
            harmonicity_values = self._compute_for_single_psd(psd.data, freqs)
            updated_metadata = {**metadata, "channels": {"dim0": list(range(len(harmonicity_values)))}}

            reduced_meta = deepcopy(updated_metadata)
            if "channels" in reduced_meta:
                del reduced_meta["channels"]

            return {
                "harmonic_spectrum": (harmonicity_values, updated_metadata),
                "max_harmonicity": (np.max(harmonicity_values), {"freq": freqs, **reduced_meta}),
                "avg_harmonicity": (np.mean(harmonicity_values), {"freq": freqs, **reduced_meta}),
            }
        elif psd.data.ndim == 2:
            freqs = metadata["channels"]["dim1"]
            if freqs[0] == 0:
                freqs[0] == 1e-8
            harmonicity_values_matrix = np.zeros(psd.data.shape)
            for i, row in enumerate(psd.data):
                harmonicity_values_matrix[i, :] = self._compute_for_single_psd(row, freqs)

            reduced_meta = deepcopy(metadata)
            if "channels" in reduced_meta and "dim1" in reduced_meta["channels"]:
                del reduced_meta["channels"]["dim1"]

            return {
                "harmonic_spectrum": (harmonicity_values_matrix, {**metadata}),
                "max_harmonicity": (np.max(harmonicity_values_matrix, axis=1), {"freq": freqs, **reduced_meta}),
                "avg_harmonicity": (np.mean(harmonicity_values_matrix, axis=1), {"freq": freqs, **reduced_meta}),
            }
        else:
            raise ValueError("Data must be either 1D or 2D")

    def _compute_for_single_psd(self, psd_values, freqs):
        power_law_remove = self.params["harmonic"]["power_law_remove"].value
        if power_law_remove:
            psd_values = self.apply_power_law_remove(freqs, psd_values, power_law_remove)
            # clip the psd_values to avoid negative values by converting them to 0
            psd_values = np.clip(psd_values, 0, None)

        metric = self.params["harmonic"]["metric"].value
        normalize = self.params["harmonic"]["normalize"].value
        n_harms = self.params["harmonic"]["n_harms"].value
        delta_lim = self.params["harmonic"]["delta_lim"].value
        min_notes = self.params["harmonic"]["min_notes"].value

        if np.array_equal(self.cached_freqs, freqs) and self.cached_metric == metric:
            harmonicity_matrix = self.cached_harmonicity_matrix
        else:
            harmonicity_matrix = self.compute_harmonicity(freqs, metric, n_harms, delta_lim, min_notes)
            # Cache the freqs and harmonicity matrix for next time
            self.cached_freqs = freqs
            self.cached_harmonicity_matrix = harmonicity_matrix
            self.cached_metric = metric  # Cache the current metric
        harmonicity_values = np.zeros(len(freqs))

        total_power = np.sum(psd_values**2)

        for i in range(len(freqs)):
            weighted_sum_harmonicity = 0
            for j in range(len(freqs)):
                if i != j:
                    weighted_sum_harmonicity += harmonicity_matrix[i, j] * (psd_values[i] * psd_values[j])

            harmonicity_values[i] = (weighted_sum_harmonicity / (2 * total_power)) if normalize else weighted_sum_harmonicity

        return harmonicity_values

    def compute_harmonicity(self, freqs, metric, n_harms, delta_lim, min_notes):
        harmonicity = np.zeros((len(freqs), len(freqs)))

        for i, f1 in enumerate(freqs):
            for j, f2 in enumerate(freqs):
                if f1 != f2:  # avoiding division by zero and self-comparison
                    if metric == "harmsim":
                        harmonicity[i, j] = self.dyad_similarity(f1 / f2)
                    elif metric == "subharm_tension":
                        _, _, subharm, _ = self.compute_subharmonic_tension(
                            [f1, f2], n_harmonics=n_harms, delta_lim=delta_lim, min_notes=min_notes
                        )
                        try:
                            harmonicity[i, j] = 1 - subharm[0]
                        except Exception:
                            harmonicity[i, j] = None
        return harmonicity
