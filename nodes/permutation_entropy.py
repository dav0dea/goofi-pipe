"""PermutationEntropy — how disordered a signal's ORDER is, from antropy.

Reads only the ranking of neighbouring samples, so amplitude scaling and slow drift do not move
it. The last axis is time and is consumed; every axis before it survives.
"""

import antropy
import numpy as np
import goofi


class PermutationEntropy(goofi.Node):
    """Permutation entropy: disorder in the ORDER of neighbouring samples."""

    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"entropy": goofi.DataType.ARRAY}
    PARAMS = {
        "permutation": {
            "order": goofi.IntParam(3, 2, 7, doc="How many samples make one ordering pattern."),
            "delay": goofi.IntParam(
                1, 1, 100, doc="Samples between the members of a pattern; raise it for a slower rhythm."
            ),
            "normalize": goofi.BoolParam(True, doc="Scale to 0..1 against the most disordered signal."),
        }
    }

    def process(self, data):
        p = self.params.permutation
        return np.apply_along_axis(
            antropy.perm_entropy,
            -1,
            np.asarray(data.data, dtype=np.float64),
            order=p.order,
            delay=p.delay,
            normalize=p.normalize,
        ).astype(np.float32)
