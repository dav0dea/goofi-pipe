"""TimbreControls — a tuning as continuous synth controls, at frame rate.

The other half of `VitalPreset`. A preset is written once and carries structure a stream cannot —
wavetables, modulation routings, the coupling analysis that costs a second to compute. This node
carries what a frame CAN: the timbre's shape, every frame, as numbers a plugin parameter binds to.

Takes a TUNING (ratios inside an octave), as `Tuning` emits them, and optionally the `amps` beside
the peaks they came from. Everything here is measured in under a millisecond, which is why it can
run at frame rate at all.

Inputs:
  input  a tuning: ratios inside an octave, as `Tuning` emits them
  amps   optional — the amplitudes beside the peaks the tuning came from

Outputs:
  partials      the ratios as frequencies over `base_freq`, in Hz
  amplitudes    per partial, 0 to 1, normalized so the loudest is 1
  weights       per partial, how consonant it is against the rest, 0 to 1
  brightness    the amplitude-weighted centroid, 0 to 1 across the partial span — the one to bind
                to a filter cutoff
  spread        how far the partials sit from a harmonic series, 0 to 1: 0 is harmonic, higher is
                bell-like. Binds to detune, unison or an inharmonic control
  harmonicity   the mean consonance of the whole set, 0 to 1

Every output is a plain number in a plain range, so binding one to a plugin param is a reference
and nothing more. The last axis is ratios and is consumed; every axis before it survives.
"""

import numpy as np
from biotuner.biotuner_utils import compute_peak_ratios
from biotuner.metrics import dyad_similarity
import goofi


class TimbreControls(goofi.Node):
    """A tuning as continuous, bindable synth controls."""

    TAGS = ["transform"]
    INPUTS = {
        "input": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "amps": goofi.InputSlot(goofi.DataType.ARRAY, required=False),
    }
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
            "spread_span": goofi.FloatParam(0.25, 0.01, 1.0, doc="The distance from harmonic that reads as spread 1."),
        }
    }

    def process(self, input, amps=None):
        p = self.params.timbre
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("TimbreControls reads a scale, not a single number")
        given = None if amps is None else np.asarray(amps.data, dtype=np.float64)

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        arows = None if given is None else given.reshape(-1, given.shape[-1])
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

            # Amplitude comes from the peaks where they are wired, else from the tilt alone; either
            # way the loudest partial is 1, so `brightness` means the same thing in both.
            if arows is not None:
                a = np.asarray([v for v in arows[i] if np.isfinite(v)], dtype=np.float64)[: ratios.size]
                a = np.power(10.0, a / 20.0) if a.size and a.min() < 0 else a  # dB from `Peaks`
                a = np.pad(a, (0, ratios.size - a.size), constant_values=0.0) if a.size < ratios.size else a
            else:
                a = np.ones(ratios.size)
            a = a * np.power(ratios, -p.tilt)
            a = a / a.max() if a.max() > 0 else a

            # How consonant each partial is against every other, as the mean of its pairs.
            w = np.array(
                [np.mean([dyad_similarity(f / g) for j, g in enumerate(ratios) if j != k]) for k, f in enumerate(ratios)]
            )
            w = np.clip(w / 100.0, 0.0, 1.0)

            centroid = float(np.sum(freqs * a) / np.sum(a)) if a.sum() > 0 else float(freqs[0])
            span = float(freqs[-1] - freqs[0])
            brightness = (centroid - freqs[0]) / span if span > 0 else 0.0
            # Distance from the nearest whole harmonic, which is what makes a set bell-like.
            harm = ratios / ratios[0]
            spread = float(np.mean(np.abs(harm - np.round(harm)))) / p.spread_span

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
