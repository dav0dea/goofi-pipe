"""Resample — the same signal at a different sample rate.

Polyphase, so the filter that stops aliasing is part of the resampling rather than a step before
it. The frame carries the new rate out; labels on the resampled axis go, because the samples they
named are gone.
"""

import numpy as np
from scipy.signal import resample_poly
import goofi


class Resample(goofi.Node):
    """Change a signal's sample rate, filtering as it goes so nothing folds back."""

    TAGS = ["transform"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PARAMS = {
        "resample": {
            "sfreq": goofi.FloatParam(250.0, 1.0, 100000.0, doc="The rate to answer at, in Hz."),
            "axis": goofi.IntParam(-1, -8, 7, doc="Which axis holds the samples. -1 is time."),
        }
    }

    def process(self, input):
        p = self.params.resample
        source = input.meta.get("sfreq")
        if not source:
            raise ValueError("this node needs a frame that carries its sample rate")
        axis = p.axis if p.axis >= 0 else p.axis + input.data.ndim
        ratio = np.gcd(int(round(p.sfreq)), int(round(source)))
        up, down = int(round(p.sfreq)) // ratio, int(round(source)) // ratio
        out = resample_poly(np.asarray(input.data, dtype=np.float64), up, down, axis=axis)
        axes = {k: v for k, v in input.meta.get("channels", {}).items() if k != f"dim{axis}"}
        return out.astype(np.float32), {**input.meta, "channels": axes, "sfreq": p.sfreq}
