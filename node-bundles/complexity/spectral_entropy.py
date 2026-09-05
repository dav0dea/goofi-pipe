"""SpectralEntropy — how flat the power spectrum is, from antropy.

A pure tone is near 0 and white noise near 1, so this reads as "how noise-like". The last axis is
time and is consumed; every axis before it survives. antropy vectorizes this one itself.
"""

import antropy
import numpy as np
import goofi


class SpectralEntropy(goofi.Node):
    """Spectral entropy: how flat the power spectrum is, tone to noise."""

    TAGS = ["analysis"]
    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"entropy": goofi.DataType.ARRAY}
    PARAMS = {
        "spectral": {
            "method": goofi.StringParam(
                "welch", ["welch", "fft"], doc="Welch averages sub-windows; fft takes the whole frame at once."
            ),
            "normalize": goofi.BoolParam(True, doc="Scale to 0..1 against a flat spectrum."),
        }
    }

    def process(self, data):
        p = self.params.spectral
        # Without a stamped rate the bins are cycles per sample. That shifts every frequency by the
        # same factor, and the measure only reads the SHAPE of the spectrum, so it is unharmed.
        sfreq = data.meta.get("sfreq") or 1.0
        return antropy.spectral_entropy(
            np.asarray(data.data, dtype=np.float64),
            sf=sfreq,
            method=p.method,
            normalize=p.normalize,
            axis=-1,
        ).astype(np.float32)
