"""SvdEntropy — how many independent components a signal's own history has, from antropy.

Embeds the signal in a delay space and reads the spread of its singular values: a rhythm needs
few, noise needs all. The last axis is time and is consumed; every axis before it survives.
"""

import antropy
import numpy as np
import goofi


class SvdEntropy(goofi.Node):
    """SVD entropy: how many independent components the signal's history spans."""

    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"entropy": goofi.DataType.ARRAY}
    PARAMS = {
        "svd": {
            "order": goofi.IntParam(3, 2, 20, doc="Dimension of the delay embedding."),
            "delay": goofi.IntParam(1, 1, 100, doc="Samples between embedding coordinates."),
            "normalize": goofi.BoolParam(True, doc="Scale to 0..1 against an even spread."),
        }
    }

    def process(self, data):
        p = self.params.svd
        return np.apply_along_axis(
            antropy.svd_entropy,
            -1,
            np.asarray(data.data, dtype=np.float64),
            order=p.order,
            delay=p.delay,
            normalize=p.normalize,
        ).astype(np.float32)
