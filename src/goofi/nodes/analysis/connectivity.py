import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class Connectivity(Node):
    """
    This node computes connectivity matrices from multichannel 2D signals. It supports both classical and biotuner-based methods for estimating pairwise relationships between channels, such as coherence, wPLI, PLV, covariance, or various harmonic similarity measures. Optionally, the output matrix can be binarized based on a threshold.

    Inputs:
    - data: 2D array signal data, typically with shape (channels, samples), and accompanying metadata.

    Outputs:
    - matrix: 2D array representing the computed connectivity between channels, with original metadata attached.
    """

    def config_input_slots():
        return {"data": DataType.ARRAY}

    def config_output_slots():
        return {
            "matrix": DataType.ARRAY,
        }

    def config_params():
        return {
            "classical": {
                "method": StringParam(
                    "wPLI",
                    options=["coherence", "imag_coherence", "wPLI", "PLI", "PLV", "covariance", "pearson", "mutual_info"],
                ),
            },
            "biotuner": {
                "method": StringParam(
                    "None", options=["None", "harmsim", "euler", "subharm_tension", "RRCi", "wPLI_crossfreq"]
                ),
                "n_peaks": IntParam(5, 1, 10, doc="Number of peaks to extract"),
                "f_min": FloatParam(2.0, 0.1, 50.0, doc="Minimum frequency"),
                "f_max": FloatParam(30.0, 1.0, 100.0, doc="Maximum frequency"),
                "precision": FloatParam(0.1, 0.01, 10.0, doc="Precision of the peak extraction in Hz"),
                "peaks_function": StringParam(
                    "EMD", options=["EMD", "fixed", "harmonic_recurrence", "EIMC"], doc="Peak extraction function"
                ),
            },
            "Adjacency": {
                "Binarize": BoolParam(False, doc="Binarize the connectivity matrix"),
                "threshold": FloatParam(0.5, 0.0, 1.0, doc="Threshold for binarization"),
            },
        }

    def process(self, data: Data):
        if data is None:
            return None

        data.data = np.squeeze(data.data)
        if data.data.ndim != 2:
            raise ValueError("Data must be 2D")

        if self.params["biotuner"]["method"].value != "None":
            matrix = compute_conn_matrix_single(
                data.data,
                data.meta["sfreq"],
                peaks_function=self.params["biotuner"]["peaks_function"].value,
                min_freq=self.params["biotuner"]["f_min"].value,
                max_freq=self.params["biotuner"]["f_max"].value,
                precision=self.params["biotuner"]["precision"].value,
                n_peaks=self.params["biotuner"]["n_peaks"].value,
                metric=self.params["biotuner"]["method"].value,
            )

        if self.params["biotuner"]["method"].value == "None":
            method = self.params["classical"]["method"].value
            matrix = compute_classical_connectivity(data.data, method)

        binarize = self.params["Adjacency"]["Binarize"].value
        threshold = self.params["Adjacency"]["threshold"].value

        if binarize:
            matrix[matrix < threshold] = 0
            matrix[matrix >= threshold] = 1
        return {"matrix": (matrix, data.meta)}


connectivity_fn = None


def compute_conn_matrix_single(
    data: np.ndarray,
    sfreq: float,
    peaks_function: str = "EMD",
    min_freq: float = 2.0,
    max_freq: float = 45.0,
    precision=0.1,
    n_peaks: int = 5,
    metric: str = "harmsim",
):
    # import the connectivity function here to avoid loading it on startup
    global connectivity_fn
    if connectivity_fn is None:
        from biotuner.harmonic_connectivity import harmonic_connectivity

        connectivity_fn = harmonic_connectivity

    # compute connectivity matrix
    bt_conn = connectivity_fn(
        sf=sfreq,
        data=data,
        peaks_function=peaks_function,
        precision=precision,
        min_freq=min_freq,
        max_freq=max_freq,
        n_peaks=n_peaks,
    )
    bt_conn.compute_harm_connectivity(metric=metric, save=False, graph=False)
    return bt_conn.conn_matrix


hilbert_fn, coherence_fn, pearsonr_fn, mutual_info_regression_fn = None, None, None, None


def compute_classical_connectivity(data, method):
    # import the connectivity function here to avoid loading it on startup
    global hilbert_fn, coherence_fn, pearsonr_fn, mutual_info_regression_fn
    if hilbert_fn is None:
        from scipy.signal import coherence, hilbert
        from scipy.stats import pearsonr
        from sklearn.feature_selection import mutual_info_regression

        hilbert_fn = hilbert
        coherence_fn = coherence
        pearsonr_fn = pearsonr
        mutual_info_regression_fn = mutual_info_regression

    n_channels, n_samples = data.shape
    matrix = np.zeros((n_channels, n_channels))

    if method == "covariance":
        matrix = np.cov(data)
        return matrix
    # TODO : optimize this shit
    for i in range(n_channels):
        for j in range(i, n_channels):  # Only compute upper diagonal
            if i == j:
                matrix[i, j] = 1  # diagonal elements are 1, for coherence/PLV or 0 for others, adjust as needed
                continue

            if method == "wPLI":
                sig1 = hilbert_fn(data[i, :])
                sig2 = hilbert_fn(data[j, :])
                imag_csd = np.imag(np.exp(1j * (np.angle(sig1) - np.angle(sig2))))
                matrix[i, j] = matrix[j, i] = np.abs(np.mean(imag_csd)) / np.mean(np.abs(imag_csd))

            elif method == "coherence":
                f, Cxy = coherence_fn(data[i, :], data[j, :])
                matrix[i, j] = matrix[j, i] = np.mean(Cxy)

            elif method == "PLI":
                sig1 = hilbert_fn(data[i, :])
                sig2 = hilbert_fn(data[j, :])
                matrix[i, j] = matrix[j, i] = np.mean(np.sign(np.angle(sig1) - np.angle(sig2)))

            elif method == "imag_coherence":
                sig1 = hilbert_fn(data[i, :])
                sig2 = hilbert_fn(data[j, :])
                matrix[i, j] = matrix[j, i] = np.mean(np.imag(np.conj(sig1) * sig2)) / (
                    np.sqrt(np.mean(np.imag(sig1) ** 2)) * np.sqrt(np.mean(np.imag(sig2) ** 2))
                )

            elif method == "PLV":
                sig1 = hilbert_fn(data[i, :])
                sig2 = hilbert_fn(data[j, :])
                matrix[i, j] = matrix[j, i] = np.abs(np.mean(np.exp(1j * (np.angle(sig1) - np.angle(sig2)))))

            elif method == "pearson":
                corr, _ = pearsonr_fn(data[i, :], data[j, :])
                matrix[i, j] = matrix[j, i] = corr

            elif method == "mutual_info":
                mutual_info = mutual_info_regression_fn(data[i, :].reshape(-1, 1), data[j, :])[0]
                matrix[i, j] = matrix[j, i] = mutual_info

    return matrix
