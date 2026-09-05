"""ZeroCrossings — how often a signal changes sign, from antropy.

The crudest frequency estimate there is, and the cheapest: a sine at f crosses zero 2f times a
second. The last axis is time and is consumed; every axis before it survives.
"""

import antropy
import numpy as np
import goofi


class ZeroCrossings(goofi.Node):
    """Zero crossings: how often the signal changes sign over the window."""

    TAGS = ["analysis"]
    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"count": goofi.DataType.ARRAY}
    PARAMS = {
        "zero": {
            "normalize": goofi.BoolParam(False, doc="Divide by the window length, for a rate per sample."),
        }
    }

    def process(self, data):
        return np.asarray(
            antropy.num_zerocross(
                np.asarray(data.data, dtype=np.float64), normalize=self.params.zero.normalize, axis=-1
            ),
            dtype=np.float32,
        )
