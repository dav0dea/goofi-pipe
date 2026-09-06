"""TuningReduction — the most consonant handful of a scale's degrees: a mode.

Takes a TUNING (ratios inside an octave), as `Tuning` emits them. A scale drawn from a spectrum
has as many degrees as it had peaks, and most of them are noise; a mode is the subset that holds
together best under one harmonicity measure.

The last axis is scale degrees and is consumed; every axis before it survives, and the output is
always `n_steps` wide — padded with NaN where the scale had fewer degrees than that to give. NaN
padding on the way in is dropped, so a `Tuning` output feeds straight in.

`n_steps` is the size of the mode you want: 5 for something pentatonic, 7 for something diatonic.
Asking for more degrees than the scale holds returns the scale.
"""

import numpy as np
from biotuner.metrics import compute_consonance, dyad_similarity, metric_denom
from biotuner.scale_construction import create_mode
import goofi

FUNCTIONS = {"harmsim": dyad_similarity, "cons": compute_consonance, "denom": metric_denom}


class TuningReduction(goofi.Node):
    """The most consonant subset of a scale, as a mode."""

    TAGS = ["analysis"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"reduced": goofi.DataType.ARRAY}
    PARAMS = {
        "mode": {
            "n_steps": goofi.IntParam(5, 2, 20, doc="Degrees the mode keeps, and the width of the output."),
            "function": goofi.StringParam(
                "harmsim",
                list(FUNCTIONS),
                doc="What decides a degree is worth keeping: `harmsim` simple ratios, `cons` consonance, "
                "`denom` the smallest denominators.",
            ),
        }
    }

    def process(self, input):
        p = self.params.mode
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("TuningReduction reads a scale, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        out = np.full((rows.shape[0], p.n_steps), np.nan)
        for i, row in enumerate(rows):
            scale = [float(v) for v in row if np.isfinite(v)]
            if len(scale) < 2:
                continue
            reduced = np.asarray(create_mode(scale, p.n_steps, FUNCTIONS[p.function]), dtype=np.float64).ravel()
            out[i, : min(reduced.size, p.n_steps)] = reduced[: p.n_steps]
        return out.reshape(lead + (p.n_steps,)).astype(np.float32)
