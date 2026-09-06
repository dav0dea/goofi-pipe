"""Tuning — the peaks of a signal read as a scale, by one of several constructions.

Takes PEAKS (frequencies in Hz), as `Peaks` emits them, and optionally the `amps` beside them.
`method` picks how a scale is built from those peaks:

  peaks_ratios        every peak over the lowest, folded into one octave. The plain reading, and
                      the only one `rebound` and `sub` apply to
  diss_curve          the minima of the dissonance curve — where two partials stop beating. The
                      one construction that USES `amps`, and it is required for it
  euler_fokker        a just scale over the primes the peaks factor into
  harmonic_tuning     the peaks read as harmonic POSITIONS of the lowest, as a harmonic series
  generator_interval  a scale stacked from one interval, ignoring the peaks entirely

The last axis is peaks and is consumed; every axis before it survives. How many degrees a row
yields depends on its peaks, so rows are padded with NaN to the widest in the batch and the nodes
downstream drop that padding again. NaN peaks are ignored, which is what lets `Peaks` feed in.

A construction that finds nothing for a row leaves that row empty rather than faulting: a scale is
a property of the signal, and a window that has none is an answer.
"""

import numpy as np
from biotuner.biotuner_utils import compute_peak_ratios, prime_factor
from biotuner.scale_construction import (
    diss_curve,
    euler_fokker_scale,
    generator_interval_tuning,
    harmonic_tuning,
)
import goofi


class Tuning(goofi.Node):
    """The ratios between a signal's peaks, as a scale inside one octave."""

    TAGS = ["analysis"]
    INPUTS = {
        "input": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "amps": goofi.InputSlot(goofi.DataType.ARRAY, required=False),
    }
    OUTPUTS = {"tuning": goofi.DataType.ARRAY}
    PARAMS = {
        "tuning": {
            "method": goofi.StringParam(
                "peaks_ratios",
                ["peaks_ratios", "diss_curve", "euler_fokker", "harmonic_tuning", "generator_interval"],
                doc="How the scale is built. `diss_curve` needs `amps` wired.",
            ),
            "octave": goofi.FloatParam(2.0, 1.1, 8.0, doc="The interval the scale folds into; 2 is the octave."),
            "rebound": goofi.BoolParam(True, doc="peaks_ratios: bring every ratio inside one octave."),
            "sub": goofi.BoolParam(False, doc="peaks_ratios: fold by subharmonics — divide down — rather than up."),
            "denom": goofi.IntParam(1000, 10, 5000, doc="diss_curve: largest denominator a minimum may name."),
            "span": goofi.FloatParam(2.0, 2.0, 8.0, doc="diss_curve: the ratio the curve runs to. 2 is one octave."),
            "interval": goofi.FloatParam(1.5, 1.01, 4.0, doc="generator_interval: the interval stacked. 1.5 is a fifth."),
            "steps": goofi.IntParam(7, 2, 53, doc="generator_interval: how many times it is stacked."),
        }
    }

    def process(self, input, amps=None):
        p = self.params.tuning
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Tuning reads a list of peaks, not a single number")
        if p.method == "diss_curve" and amps is None:
            raise ValueError("`diss_curve` needs the `amps` input wired — `Peaks` emits it beside the peaks")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        arows = None
        if amps is not None:
            av = np.asarray(amps.data, dtype=np.float64)
            arows = av.reshape(-1, av.shape[-1]) if av.ndim else None

        found = []
        for i, row in enumerate(rows):
            ramps = None if arows is None else arows[i % arows.shape[0]]
            found.append(self._scale(row, ramps, p))

        width = max((r.size for r in found), default=0) or 1
        out = np.full((rows.shape[0], width), np.nan)
        for i, r in enumerate(found):
            out[i, : r.size] = r
        return out.reshape(lead + (width,)).astype(np.float32)

    def _scale(self, row, row_amps, p):
        peaks = np.asarray([v for v in row if np.isfinite(v)], dtype=np.float64)
        # A scale is a set of INTERVALS, so one peak makes none — except for the construction that
        # does not read the peaks at all.
        if p.method == "generator_interval":
            return np.asarray(generator_interval_tuning(interval=p.interval, steps=p.steps, octave=p.octave)[0])
        if peaks.size < 2 or peaks[0] <= 0:
            return np.empty(0)

        if p.method == "peaks_ratios":
            got = compute_peak_ratios(peaks.tolist(), rebound=p.rebound, octave=p.octave, sub=p.sub)
        elif p.method == "diss_curve":
            amp = np.asarray([v for v in row_amps if np.isfinite(v)], dtype=np.float64)[: peaks.size]
            if amp.size < peaks.size:
                amp = np.pad(amp, (0, peaks.size - amp.size), constant_values=0.0)
            got = diss_curve(peaks.tolist(), amp.tolist(), denom=p.denom, max_ratio=p.span,
                             euler_comp=False, method="min", plot=False)[2]
        elif p.method == "euler_fokker":
            whole = [int(v) for v in peaks if v >= 1.0]
            got = euler_fokker_scale(prime_factor(whole), octave=p.octave) if whole else []
        else:
            got = harmonic_tuning(sorted({max(1, int(round(v / peaks[0]))) for v in peaks}), octave=p.octave)

        return np.asarray(got, dtype=np.float64).ravel()
