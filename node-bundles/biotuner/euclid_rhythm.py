"""EuclidRhythm — a scale read as rhythms: every interval becomes a euclidean pattern.

Takes a TUNING (ratios), as `Tuning` emits them, and turns each into a euclidean rhythm — the
pattern that spreads `pulses` onsets over `steps` as evenly as whole numbers allow. The ratio is
the source of both: a 3/2 becomes three pulses over two steps, so an interval a scale is built
from and the rhythm beside it are the same number read at two speeds.

  patterns   one row per rhythm, 1 where an onset falls and 0 where none does
  steps      how many steps each row spans, so a row can be read without counting its padding

`mode` decides which rhythms are kept. `normal` takes the numerator over the denominator, `full`
takes both that and its inverse. `consonant` is different in kind: it walks the scale down
`descend` octaves and keeps only the pairs whose interval is consonant to within `tolerance`,
which is far fewer rhythms and all of them related.

Rows are padded with NaN to the longest, as the rest of the bundle pads: the shape a viewer sees
never depends on the data, and `steps` says where each row truly ends. Every axis before the last
survives, so a scale per channel gives rhythms per channel.
"""

import numpy as np
from biotuner.rhythm_construction import consonant_euclid, scale2euclid
import goofi


class EuclidRhythm(goofi.Node):
    """A scale's intervals as euclidean rhythms."""

    TAGS = ["transform", "music"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"patterns": goofi.DataType.ARRAY, "steps": goofi.DataType.ARRAY}
    PARAMS = {
        "euclid": {
            "mode": goofi.StringParam(
                "normal", ["normal", "full", "consonant"], doc="Which rhythms a ratio yields. `consonant` filters instead."
            ),
            "maxDenom": goofi.IntParam(10, 2, 64, doc="Largest denominator a ratio may be approximated by."),
            "descend": goofi.IntParam(2, 1, 8, doc="consonant: octaves the scale is walked down before pairing."),
            "tolerance": goofi.FloatParam(0.1, 0.001, 1.0, doc="consonant: how far from consonant a pair may sit."),
            "finalDenom": goofi.IntParam(16, 2, 64, doc="consonant: largest denominator a kept rhythm may have."),
        }
    }

    def process(self, input):
        p = self.params.euclid
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("EuclidRhythm reads a scale, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        found = [self._rhythms(row, p) for row in rows]

        # Two paddings, because a row yields a VARYING number of rhythms of VARYING length.
        deep = max((len(f) for f in found), default=0) or 1
        wide = max((len(pat) for f in found for pat in f), default=0) or 1
        pats = np.full((rows.shape[0], deep, wide), np.nan)
        spans = np.full((rows.shape[0], deep), np.nan)
        for i, f in enumerate(found):
            for k, pat in enumerate(f):
                pats[i, k, : len(pat)] = pat
                spans[i, k] = len(pat)

        return {
            "patterns": pats.reshape(lead + (deep, wide)).astype(np.float32),
            "steps": spans.reshape(lead + (deep,)).astype(np.float32),
        }

    def _rhythms(self, row, p):
        scale = [float(v) for v in row if np.isfinite(v) and v > 0]
        # A rhythm comes from an INTERVAL, so one degree makes none.
        if len(scale) < 2:
            return []
        if p.mode == "consonant":
            kept, _steps = consonant_euclid(
                scale, n_steps_down=p.descend, limit_denom=64, limit_cons=p.tolerance, limit_denom_final=p.finalDenom
            )
        else:
            kept = scale2euclid(scale, max_denom=p.maxDenom, mode=p.mode)
        # A one-step pattern is a ratio that reduced to nothing; it is not a rhythm.
        return [list(pat) for pat in kept if pat is not None and len(pat) > 1]
