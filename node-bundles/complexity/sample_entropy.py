"""SampleEntropy — how unpredictable a signal is, from antropy.

Counts how often a short pattern that recurred keeps recurring one sample longer; a signal that
is regular scores near 0. The last axis is time and is consumed; every axis before it survives.
"""

import antropy
import numpy as np
import goofi


class SampleEntropy(goofi.Node):
    """Sample entropy: how rarely a pattern that recurred keeps recurring."""

    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"entropy": goofi.DataType.ARRAY}
    PARAMS = {
        "sample": {
            "order": goofi.IntParam(2, 1, 10, doc="How many samples make one pattern."),
            "metric": goofi.StringParam(
                "chebyshev", ["chebyshev", "euclidean"], doc="How far apart two patterns may be to match."
            ),
            "approximate": goofi.BoolParam(
                False, doc="Count a pattern as matching itself: approximate entropy, the older biased form."
            ),
        }
    }

    def process(self, data):
        p = self.params.sample
        measure = antropy.app_entropy if p.approximate else antropy.sample_entropy
        return np.apply_along_axis(
            measure, -1, np.asarray(data.data, dtype=np.float64), order=p.order, metric=p.metric
        ).astype(np.float32)
