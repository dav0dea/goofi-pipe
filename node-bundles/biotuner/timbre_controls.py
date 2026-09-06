"""TimbreControls — a tuning as continuous synth controls, at frame rate.

The other half of `VitalPreset`. A preset is written once and carries structure a stream cannot —
wavetables, modulation routings, the coupling analysis that costs a second to compute. This node
carries what a frame CAN: the timbre's shape, every frame, as numbers a plugin parameter binds to.

Takes a TUNING (ratios inside an octave), as `Tuning` emits them. Everything here is measured in
under a millisecond, which is why it can run at frame rate at all.

It takes NO amplitudes, and that is a decision rather than an omission. `compute_peak_ratios`
answers every PAIRWISE ratio, deduplicated and folded — five peaks give ten degrees — so degree
`k` stands in no relation to peak `k` and an amplitude cannot be aligned to it. The information
needed to align them does not survive the tuning, so it cannot be repaired here: an amplitude
belongs to a peak, and weighting it belongs where the peaks still exist. `tilt` shapes the
partials instead.


Every output is a plain number in a plain range, so binding one to a plugin param is a reference
and nothing more. The last axis is ratios and is consumed; every axis before it survives.
"""

from fractions import Fraction

import numpy as np
from biotuner.biotuner_utils import compute_peak_ratios
from biotuner.metrics import dyad_similarity
import goofi


class TimbreControls(goofi.Node):
    """A tuning as continuous, bindable synth controls.

    Inputs:
      input  a tuning: ratios inside an octave, as `Tuning` emits them

    Outputs:
      partials      the ratios as frequencies over `base_freq`, in Hz
      amplitudes    per partial, 0 to 1, normalized so the loudest is 1
      weights       per partial, how consonant it is against the rest, 0 to 1
      brightness    the amplitude-weighted centroid, 0 to 1 across the partial span — the one to bind
                    to a filter cutoff
      spread        how far the degrees sit from simple just ratios, in cents against `spread_span`,
                    0 to 1: a just scale is 0, a tempered or irrational one higher. Binds to
                    detune, unison or an inharmonic control
      harmonicity   the mean consonance of the whole set, 0 to 1
    """

    TAGS = ["transform"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {
        "partials": goofi.DataType.ARRAY,
        "amplitudes": goofi.DataType.ARRAY,
        "weights": goofi.DataType.ARRAY,
        "brightness": goofi.DataType.ARRAY,
        "spread": goofi.DataType.ARRAY,
        "harmonicity": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "timbre": {
            "base_freq": goofi.FloatParam(220.0, 20.0, 2000.0, doc="The frequency ratio 1 sits at, in Hz."),
            "tilt": goofi.FloatParam(0.0, -2.0, 2.0, doc="Amplitude rolloff per partial: above 0 favours the low ones."),
            "spread_span": goofi.FloatParam(25.0, 1.0, 200.0, doc="Cents away from just that reads as spread 1."),
            "justLimit": goofi.IntParam(8, 2, 32, doc="Largest denominator a degree may be called just by."),
        }
    }

    @staticmethod
    def _cents_from_just(ratios, limit):
        """Mean distance, in cents, from each degree to the simplest just ratio near it."""
        away = []
        for r in ratios:
            near = Fraction(float(r)).limit_denominator(int(limit))
            away.append(abs(1200.0 * np.log2(float(r) / float(near))) if float(near) > 0 else 0.0)
        return float(np.mean(away)) if away else 0.0

    def process(self, input):
        p = self.params.timbre
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("TimbreControls reads a scale, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        width = max(int(np.isfinite(r).sum()) for r in rows) or 1
        part = np.full((rows.shape[0], width), np.nan)
        amp = np.full((rows.shape[0], width), np.nan)
        wgt = np.full((rows.shape[0], width), np.nan)
        scal = np.full((rows.shape[0], 3), np.nan)

        for i, row in enumerate(rows):
            ratios = np.asarray([v for v in row if np.isfinite(v)], dtype=np.float64)
            if ratios.size < 2:
                continue
            freqs = ratios * p.base_freq

            # `tilt` alone shapes the amplitudes, and the loudest partial is 1.
            a = np.power(ratios, -p.tilt)
            a = a / a.max() if a.max() > 0 else a

            # How consonant each partial is against every other, as the mean of its pairs.
            w = np.array(
                [np.mean([dyad_similarity(f / g) for j, g in enumerate(ratios) if j != k]) for k, f in enumerate(ratios)]
            )
            w = np.clip(w / 100.0, 0.0, 1.0)

            centroid = float(np.sum(freqs * a) / np.sum(a)) if a.sum() > 0 else float(freqs[0])
            span = float(freqs[-1] - freqs[0])
            brightness = (centroid - freqs[0]) / span if span > 0 else 0.0
            # Distance from the nearest SIMPLE JUST ratio. Measuring against the nearest whole
            # number cannot work on a scale: every ratio is folded into `[1, 2)`, so the distance
            # is to 1 or to 2, which an evenly spread scale maximises by construction — it rated a
            # just major scale as more bell-like than a deliberately inharmonic one.
            spread = self._cents_from_just(ratios, p.justLimit) / p.spread_span

            part[i, : ratios.size] = freqs
            amp[i, : a.size] = a
            wgt[i, : w.size] = w
            scal[i] = (np.clip(brightness, 0.0, 1.0), np.clip(spread, 0.0, 1.0), float(np.mean(w)))

        f32 = lambda v, shape: v.reshape(shape).astype(np.float32)
        s = scal.reshape(lead + (3,)).astype(np.float32)
        return {
            "partials": f32(part, lead + (width,)),
            "amplitudes": f32(amp, lead + (width,)),
            "weights": f32(wgt, lead + (width,)),
            "brightness": s[..., 0],
            "spread": s[..., 1],
            "harmonicity": s[..., 2],
        }
