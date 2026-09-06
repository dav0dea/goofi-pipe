"""Polyrhythm — a scale played as several rhythms at once, one voice per degree.

Takes a TUNING (ratios), as `Tuning` emits them, and gives every degree its own cycle. The ratios
are the tempo relations between the voices, so a scale and the polyrhythm it makes are the same
intervals heard at two speeds — one as pitch, one as pulse. Every voice shares one grid, whose
length is the lowest common multiple of their cycles, and where their onsets land together is a
`coincidence`: the rhythmic reading of consonance.

Inputs:
  input  a tuning: ratios inside an octave, as `Tuning` emits them

Outputs:
  voices        one row per degree, 1 where an onset falls and 0 where none does
  coincidences  per grid position, how many voices strike at once
  labels        what each voice is, in the construction's own words
  cycle         how many positions the grid holds, so a player knows the loop length

`method` picks the construction. `euclid` gives each voice a euclidean pattern over its own step
count. `iso` spaces each voice evenly across the whole grid, which makes a longer cycle and a
plainer one. `harmonic` derives the cycles from the ratios directly.

`cycleCap` is the one that keeps this playable. The grid is a lowest common multiple, so it grows
without bound on a real tuning — a measured one wanted 18018 positions, which is a frame to carry
rather than a bar to hear — and the denominator is lowered until the grid fits. `cycle` reports
what it settled on, so nothing about that is hidden.

Unlike the rest of the bundle this pads nothing: every voice already shares the grid. Every axis
before the last survives, and `labels` joins its rows with `; ` because a STRING carries one.
"""

import math

import numpy as np
from biotuner.biotuner_utils import scale2frac
from biotuner.rhythm_construction import (
    scale2polyrhythm,
    scale2polyrhythm_harmonic,
    scale2polyrhythm_iso,
)
import goofi


def fitting_denom(scale, want, cap):
    """The largest denominator whose grid cannot outgrow `cap`, walking down from `want`.

    The grid is the lowest common multiple of terms the construction keeps from `scale2frac`, so
    the multiple of ALL of them bounds it whatever is kept — a bound rather than a copy of the
    filter, which is what keeps this true when biotuner's own changes.
    """
    for denom in range(int(want), 1, -1):
        _frac, nums, denoms = scale2frac(scale, maxdenom=denom)
        terms = [int(t) for t in list(nums) + list(denoms) if 0 < int(t) <= denom]
        if not terms or math.lcm(*terms) <= cap:
            return denom
    return 2


class Polyrhythm(goofi.Node):
    """A scale's degrees as simultaneous cycles on one grid."""

    TAGS = ["transform", "music"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {
        "voices": goofi.DataType.ARRAY,
        "coincidences": goofi.DataType.ARRAY,
        "labels": goofi.DataType.STRING,
        "cycle": goofi.DataType.ARRAY,
    }
    PARAMS = {
        "poly": {
            "method": goofi.StringParam(
                "euclid", ["euclid", "iso", "harmonic"], doc="How each degree becomes a cycle."
            ),
            "maxDenom": goofi.IntParam(16, 2, 64, doc="euclid, iso: largest denominator a ratio may be approximated by."),
            "harmonics": goofi.IntParam(4, 1, 16, doc="harmonic: how many harmonics each degree contributes."),
            "cycleCap": goofi.IntParam(64, 4, 2048, doc="Longest grid allowed. The denominator drops until it fits."),
        }
    }

    def process(self, input):
        p = self.params.poly
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Polyrhythm reads a scale, not a single number")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        made = [self._grid(row, p) for row in rows]

        deep = max((v.shape[0] for v, _, _ in made), default=0) or 1
        wide = max((v.shape[1] for v, _, _ in made), default=0) or 1
        voices = np.full((rows.shape[0], deep, wide), np.nan)
        coinc = np.full((rows.shape[0], wide), np.nan)
        cycles = np.zeros(rows.shape[0])
        words = []
        for i, (v, c, label) in enumerate(made):
            voices[i, : v.shape[0], : v.shape[1]] = v
            coinc[i, : c.size] = c
            cycles[i] = v.shape[1]
            words.append(label)

        return {
            "voices": voices.reshape(lead + (deep, wide)).astype(np.float32),
            "coincidences": coinc.reshape(lead + (wide,)).astype(np.float32),
            "labels": "; ".join(w for w in words if w),
            "cycle": cycles.reshape(lead + (1,)).astype(np.float32),
        }

    def _grid(self, row, p):
        scale = [float(v) for v in row if np.isfinite(v) and v > 0]
        # Voices are the RELATIONS between degrees, so one degree makes no polyrhythm.
        if len(scale) < 2:
            return np.zeros((1, 1)), np.zeros(1), ""
        if p.method == "harmonic":
            got = scale2polyrhythm_harmonic(scale, n_harmonics=p.harmonics, lcm_cap=p.cycleCap)
        else:
            # The grid is a lowest common multiple, so it grows without bound on a real tuning: a
            # measured one gave denominators [1,16,16,16,13,12,10,8,6,16] and an 18018-step grid,
            # which is a frame the subprocess dies carrying rather than a rhythm anyone hears.
            denom = fitting_denom(scale, p.maxDenom, p.cycleCap)
            if p.method == "iso":
                got = scale2polyrhythm_iso(scale, max_denom=denom)
            else:
                got = scale2polyrhythm(scale, max_denom=denom)

        voices, coincidences, _positions, labels, _lcm = got
        if not voices:
            return np.zeros((1, 1)), np.zeros(1), ""
        return (
            np.asarray(voices, dtype=np.float64),
            np.asarray(coincidences, dtype=np.float64).ravel(),
            ",".join(str(v) for v in labels),
        )
