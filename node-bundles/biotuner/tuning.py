"""Tuning — the peaks of a signal read as a scale: their ratios, folded into one octave.

Takes PEAKS (frequencies in Hz), as `Peaks` emits them. Every peak is divided by the lowest and
rebounded into `[1, octave]`, so a spectrum becomes a set of intervals a scale can be built from.
The ratios come back sorted, with the octave closing the scale — `1.5` is a perfect fifth, `2.0`
the octave itself.

The last axis is peaks and is consumed; every axis before it survives. How many degrees a row
yields depends on how many peaks it had, so rows are padded with NaN to the widest in the batch
and `TuningMatrix` and `TuningReduction` drop that padding again. NaN peaks are ignored, which is
what lets a `Peaks` output feed straight in.
"""

import numpy as np
from biotuner.biotuner_utils import compute_peak_ratios
import goofi


class Tuning(goofi.Node):
    """The ratios between a signal's peaks, as a scale inside one octave."""

    TAGS = ["analysis"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"tuning": goofi.DataType.ARRAY}
    PARAMS = {
        "tuning": {
            "octave": goofi.FloatParam(2.0, 1.1, 8.0, doc="The interval the scale folds into; 2 is the octave."),
            "rebound": goofi.BoolParam(True, doc="Bring every ratio inside one octave. Off keeps them as found."),
            "sub": goofi.BoolParam(False, doc="Fold by subharmonics — divide down — rather than harmonics."),
        }
    }

    def process(self, input):
        p = self.params.tuning
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Tuning reads a list of peaks, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        found = []
        for row in rows:
            peaks = [float(v) for v in row if np.isfinite(v)]
            # A scale is a set of INTERVALS, so one peak makes none.
            found.append(
                np.asarray(compute_peak_ratios(peaks, rebound=p.rebound, octave=p.octave, sub=p.sub), dtype=np.float64)
                if len(peaks) >= 2
                else np.empty(0)
            )

        width = max((r.size for r in found), default=0) or 1
        out = np.full((rows.shape[0], width), np.nan)
        for i, r in enumerate(found):
            out[i, : r.size] = r
        return out.reshape(lead + (width,)).astype(np.float32)
