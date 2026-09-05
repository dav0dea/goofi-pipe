"""LempelZiv — LZ76 complexity, the EEG regularity measure, from antropy.

The last axis is time and it is what the measure consumes: `[C, T]` in, `[C]` out. Every axis
before it survives, because a complexity per channel is still a value per channel.
"""

import antropy
import numpy as np
import goofi


class LempelZiv(goofi.Node):
    """LZ76 complexity: how much a signal repeats itself."""

    TAGS = ["analysis"]
    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"complexity": goofi.DataType.ARRAY}
    PARAMS = {
        "lz": {
            "threshold": goofi.StringParam(
                "median",
                ["median", "mean"],
                doc="What each sample is called high or low against before the sequence is counted.",
            ),
            "normalize": goofi.BoolParam(
                True, doc="Divide by the complexity a random sequence of the same length would have."
            ),
        }
    }

    def process(self, data):
        p = self.params.lz

        def lz(x):
            level = np.median(x) if p.threshold == "median" else np.mean(x)
            return antropy.lziv_complexity((x > level).astype(int), normalize=p.normalize)

        x = np.asarray(data.data, dtype=np.float64)
        return np.apply_along_axis(lz, -1, x).astype(np.float32)
