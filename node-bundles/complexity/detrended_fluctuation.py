"""DetrendedFluctuation — the DFA exponent, from antropy.

Says how a signal's fluctuation grows with the window it is measured over: near 0.5 is noise with
no memory, near 1.0 is long-range correlated, above 1.5 is a random walk. The last axis is time and
is consumed; every axis before it survives.
"""

import antropy
import numpy as np
import goofi


class DetrendedFluctuation(goofi.Node):
    """DFA exponent: how far a signal remembers its own past."""

    TAGS = ["analysis"]
    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"exponent": goofi.DataType.ARRAY}

    def process(self, data):
        return np.apply_along_axis(
            antropy.detrended_fluctuation, -1, np.asarray(data.data, dtype=np.float64)
        ).astype(np.float32)
