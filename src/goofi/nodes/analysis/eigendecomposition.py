from copy import deepcopy

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, StringParam


class EigenDecomposition(Node):
    """
    Performs eigen decomposition on a 2D matrix input. Optionally, the node can compute the (unnormalized or normalized) Laplacian of the matrix before decomposition. Different algorithms for eigenvalue and eigenvector computation are supported. The node returns both the eigenvalues and eigenvectors of the (transformed) matrix, with outputs ordered as specified and, if needed, with a consistent sign orientation.

    Inputs:
    - matrix: A 2D array representing the matrix to decompose.

    Outputs:
    - eigenvalues: Array containing the eigenvalues of the input (or Laplacian) matrix.
    - eigenvectors: 2D array where each column is an eigenvector corresponding to an eigenvalue.
    """

    def config_input_slots():
        return {"matrix": DataType.ARRAY}

    def config_output_slots():
        return {
            "eigenvalues": DataType.ARRAY,
            "eigenvectors": DataType.ARRAY,
        }

    def config_params():
        return {
            "Eigen": {
                "laplacian": StringParam("none", options=["none", "unnormalized", "normalized"]),
                "method": StringParam("eig", options=["eig", "eigh", "eigh_general"]),
                "sign_shift": BoolParam(False),
                "order": StringParam("descending", options=["descending", "ascending"]),
            }
        }

    def process(self, matrix: Data):
        if matrix is None:
            return None

        matrix_data = np.squeeze(matrix.data)
        if matrix_data.ndim != 2:
            raise ValueError("Matrix must be 2D")

        if self.params.Eigen.laplacian.value == "unnormalized":
            # Compute the Laplacian of the connectivity matrix
            matrix_data = laplacian(matrix_data, normed=False)
        if self.params.Eigen.laplacian.value == "normalized":
            matrix_data = laplacian(matrix_data, normed=True)

        method = self.params.Eigen.method.value

        if method == "eigh":
            eigenvalues, eigenvectors = np.linalg.eigh(matrix_data)
        if method == "eigh_general":
            eigenvalues, eigenvectors = eigh(matrix_data)
        if method == "eig":
            eigenvalues, eigenvectors = np.linalg.eig(matrix_data)

        if self.params.Eigen.order.value == "descending":
            idx = eigenvalues.argsort()[::-1]
        elif self.params.Eigen.order.value == "ascending":
            idx = eigenvalues.argsort()
        # reordering eigenvalues and eigenvectors and channel names (which are strings)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.params.Eigen.sign_shift.value:
            signs = np.sign(np.sum(eigenvectors, axis=0))
            eigenvectors *= signs

        if "dim0" in matrix.meta["channels"]:
            matrix.meta["channels"]["dim0"] = [matrix.meta["channels"]["dim0"][i] for i in idx]

        copied_meta = deepcopy(matrix.meta)
        del copied_meta["channels"]

        return {
            "eigenvalues": (np.array(eigenvalues).astype(np.float32), copied_meta),
            "eigenvectors": (np.array(eigenvectors).astype(np.float32), matrix.meta),
        }
