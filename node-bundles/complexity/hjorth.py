"""Hjorth — mobility and complexity, the two derivative-based EEG descriptors, from antropy.

Mobility is the signal's mean frequency in units of the sample rate; complexity is how far its
shape is from a pure sine, which scores exactly 1. The last axis is time and is consumed; every
axis before it survives.
"""

import antropy
import numpy as np
import goofi


class Hjorth(goofi.Node):
    """Hjorth mobility and complexity: mean frequency, and distance from a sine."""

    TAGS = ["analysis"]
    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"mobility": goofi.DataType.ARRAY, "complexity": goofi.DataType.ARRAY}

    def process(self, data):
        mobility, complexity = antropy.hjorth_params(np.asarray(data.data, dtype=np.float64), axis=-1)
        return {
            "mobility": np.asarray(mobility, dtype=np.float32),
            "complexity": np.asarray(complexity, dtype=np.float32),
        }
