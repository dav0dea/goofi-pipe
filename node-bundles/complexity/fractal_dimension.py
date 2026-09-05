"""FractalDimension — how much a signal's trace fills the plane, from antropy.

A smooth curve scores near 1 and a trace that fills its box approaches 2. Three estimators
under one `method`: Petrosian counts turns, Katz measures path length, Higuchi reads the length
at several scales and is the slowest. The last axis is time and is consumed; every axis before
it survives.
"""

import antropy
import numpy as np
import goofi


class FractalDimension(goofi.Node):
    """Fractal dimension: how much the trace fills the plane, 1 (smooth) to 2 (filling)."""

    TAGS = ["analysis"]
    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"dimension": goofi.DataType.ARRAY}
    PARAMS = {
        "fractal": {
            "method": goofi.StringParam(
                "petrosian",
                ["petrosian", "katz", "higuchi"],
                doc="Petrosian counts turns, Katz measures path length, Higuchi reads several scales.",
            ),
            "kmax": goofi.IntParam(10, 2, 100, doc="Higuchi only: the coarsest scale read."),
        }
    }

    def process(self, data):
        p = self.params.fractal
        x = np.asarray(data.data, dtype=np.float64)
        if p.method == "higuchi":
            return np.apply_along_axis(antropy.higuchi_fd, -1, x, kmax=p.kmax).astype(np.float32)
        measure = antropy.katz_fd if p.method == "katz" else antropy.petrosian_fd
        return np.asarray(measure(x, axis=-1), dtype=np.float32)
