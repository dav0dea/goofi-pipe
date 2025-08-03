from itertools import groupby

import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam


class Recurrence(Node):
    """
    Computes the recurrence matrix and several recurrence quantification analysis (RQA) metrics for a given input array. The recurrence matrix represents recurrent states in the data based on pairwise distances, and the RQA metrics quantify aspects of the recurrence structure, including recurrence rate, determinism, and laminarity.

    Inputs:
    - input_array: N-dimensional array of data points to analyze for recurrence structures.

    Outputs:
    - recurrence_matrix: Matrix indicating where recurrences occur in the input data.
    - RR: Array containing the calculated recurrence rate.
    - DET: Array containing the calculated determinism.
    - LAM: Array containing the calculated laminarity.
    """

    def config_input_slots():
        return {"input_array": DataType.ARRAY}

    def config_output_slots():
        return {"recurrence_matrix": DataType.ARRAY, "RR": DataType.ARRAY, "DET": DataType.ARRAY, "LAM": DataType.ARRAY}

    def config_params():
        return {
            "recurrence": {
                "relative_threshold": FloatParam(
                    0.1,
                    0.0,
                    1.0,
                    doc="Relative threshold for binarizing the distance matrix as a fraction of the maximum distance",
                ),
                "binarize": BoolParam(True, doc="Flag to determine if distance matrix should be binarized"),
                "min_diag_length": IntParam(
                    2, 1, 100, doc="Minimum length of diagonal lines to be considered for DET and LAM calculation"
                ),
            }
        }

    def setup(self):
        from sklearn.metrics import pairwise_distances

        self.pairwise_distances = pairwise_distances

    def process(self, input_array: Data):
        if input_array is None:
            return None

        # Get the parameters
        relative_threshold = self.params["recurrence"]["relative_threshold"].value
        binarize = self.params["recurrence"]["binarize"].value

        # Check the shape of the array
        data = input_array.data
        if len(data.shape) == 1:
            # Reshape 1D array to 2D column vector
            data = data[:, np.newaxis]

        if data.shape[0] < 2:
            raise ValueError("Not enough data points to compute recurrence matrix.")

        # Compute pairwise distance
        distance_matrix = self.pairwise_distances(data)

        # Calculate the actual threshold
        threshold = relative_threshold * np.max(distance_matrix)

        # Binarize the distance matrix if required
        if binarize:
            recurrence_matrix = (distance_matrix <= threshold).astype(int)
        else:
            recurrence_matrix = distance_matrix

        # Calculate RQA metrics
        RR = np.mean(recurrence_matrix)  # Recurrence Rate
        # Get all diagonal lengths using the new method
        diagonal_lengths = self.get_diagonal_lengths(recurrence_matrix)
        DET = sum([ln**2 for ln in diagonal_lengths]) / float(recurrence_matrix.size)  # Determinism
        vertical_lengths = [
            len(list(group))
            for i in range(recurrence_matrix.shape[0])
            for key, group in groupby(recurrence_matrix[i, :])
            if key
        ]
        LAM = sum([ln**2 for ln in vertical_lengths]) / float(recurrence_matrix.size)  # Laminarity

        return {
            "recurrence_matrix": (recurrence_matrix, input_array.meta),
            "RR": (np.array(RR), {}),
            "DET": (np.array(DET), {}),
            "LAM": (np.array(LAM), {}),
        }

    def get_diagonal_lengths(self, matrix):
        N = matrix.shape[0]
        lengths = []
        for d in range(-N + 2, N - 1):  # Loop over all diagonals
            diag = np.diagonal(matrix, offset=d)
            for key, group in groupby(diag):
                if key:
                    length = len(list(group))
                    if length >= self.params["recurrence"]["min_diag_length"].value:
                        lengths.append(length)
        return lengths
