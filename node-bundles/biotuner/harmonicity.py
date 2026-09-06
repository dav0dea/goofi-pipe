"""Harmonicity — how consonant a set of peaks is, by four of biotuner's measures.

Takes PEAKS (frequencies in Hz), as `Peaks` emits them — not a signal and not a spectrum. NaN
padding is dropped per row, so a channel that found fewer peaks is still measured on what it has.

Each output is ONE number per channel, which is what makes it something a param can follow:
`[C, n_peaks]` in gives `[C]` out, and a single row gives a single value.


`harmsim` and `cons` rise together; `tenney` and `subharmTension` run the other way, so pairing
one of each is the usual way to drive two params in opposition.
"""

import numpy as np
from biotuner.metrics import (
    compute_subharmonic_tension,
    consonance_peaks,
    peaks_to_harmsim,
    tenneyHeight,
)
import goofi


class Harmonicity(goofi.Node):
    """Harmonic similarity, Tenney height, consonance and subharmonic tension.

    Inputs:
      input  peaks in Hz, as `Peaks` emits them

    Outputs:
      harmsim          0..100, higher is more consonant. The mean harmonic similarity of every pair:
                       how nearly the peaks form a simple whole-number ratio.
      tenney           Tenney height, higher is MORE complex. The log of the ratio's numerator times
                       its denominator, so it grows as the fractions get uglier.
      cons             0..1, higher is more consonant. The mean consonance of the peak pairs that
                       pass `cons_limit`.
      subharmTension   Higher is more tense. How badly the peaks fail to share a common subharmonic
                       within `delta_lim`.
    """

    TAGS = ["analysis"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {
        "harmsim": goofi.DataType.ARRAY,
        "tenney": goofi.DataType.ARRAY,
        "cons": goofi.DataType.ARRAY,
        "subharmTension": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "harmonicity": {
            "n_harm": goofi.IntParam(3, 1, 10, doc="Harmonics compared when weighing subharmonic tension."),
            "delta_lim": goofi.IntParam(250, 1, 300, doc="Widest subharmonic beat still counted, in ms."),
            "min_notes": goofi.IntParam(2, 2, 10, doc="Peaks that must agree before a subharmonic counts."),
            "cons_limit": goofi.FloatParam(0.1, 0.001, 1.0, doc="Smallest interval still called consonant."),
        }
    }

    def process(self, input):
        p = self.params.harmonicity
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Harmonicity reads a list of peaks, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        out = np.full((rows.shape[0], 4), np.nan)
        for i, row in enumerate(rows):
            chord = [float(v) for v in row if np.isfinite(v)]
            # Harmonicity is a RELATION: one peak has nothing to be consonant with, and the row
            # answers NaN rather than a number that would read as "perfectly consonant".
            if len(chord) < 2:
                continue
            tension = compute_subharmonic_tension(chord, p.n_harm, p.delta_lim, min_notes=p.min_notes)[2]
            tension = np.asarray(tension, dtype=np.float64).ravel()
            out[i] = (
                float(np.mean(peaks_to_harmsim(chord))),
                float(tenneyHeight(chord)),
                float(consonance_peaks(chord, p.cons_limit)[3]),
                float(tension[0]) if tension.size else np.nan,
            )

        cols = out.reshape(lead + (4,)).astype(np.float32)
        return {
            "harmsim": cols[..., 0],
            "tenney": cols[..., 1],
            "cons": cols[..., 2],
            "subharmTension": cols[..., 3],
        }
