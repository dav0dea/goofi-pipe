"""Eigen — the axes of a square symmetric matrix, strongest first.

A connectivity matrix says how every pair of channels relates; its eigenvectors say which
combinations of channels move together, and the eigenvalues how much each one carries. The
Laplacian modes read the matrix as a graph instead, where the smallest values are the ones that
say how it splits.
"""

import numpy as np
import goofi


class Eigen(goofi.Node):
    """The eigenvalues and eigenvectors of a square symmetric matrix, or of its Laplacian."""

    TAGS = ["analysis", "connectivity"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"values": goofi.DataType.ARRAY, "vectors": goofi.DataType.ARRAY}
    PARAMS = {
        "eigen": {
            "laplacian": goofi.StringParam(
                "none",
                options=["none", "unnormalized", "normalized"],
                doc="Read the matrix as it is, or as the graph Laplacian built from it.",
            ),
            "order": goofi.StringParam(
                "descending",
                options=["descending", "ascending"],
                doc="Which end comes first. A Laplacian is usually read the other way round.",
            ),
        }
    }

    def process(self, input):
        p = self.params.eigen
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError(f"needs a square matrix, got {list(x.shape)}")
        x = (x + x.T) / 2

        if p.laplacian != "none":
            degree = np.abs(x).sum(axis=1)
            x = np.diag(degree) - x
            if p.laplacian == "normalized":
                scale = 1 / np.sqrt(np.where(degree > 0, degree, 1))
                x = x * scale[:, None] * scale[None, :]

        values, vectors = np.linalg.eigh(x)
        if p.order == "descending":
            values, vectors = values[::-1], vectors[:, ::-1]

        axes = input.meta.get("channels", {})
        return {
            "values": (values.astype(np.float32), {}),
            "vectors": (np.ascontiguousarray(vectors, dtype=np.float32), {"channels": axes}),
        }
