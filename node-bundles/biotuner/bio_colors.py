"""BioColors — a signal's peaks, or the scale they make, as a palette.

Takes PEAKS in Hz (as `Peaks` emits them) or a TUNING of ratios, and answers a colour per degree.
The mapping is not decoration: biotuner places each partial in a perceptual space — OKLCh, where a
step of the same size looks the same size — and `method` chooses what the hue is a picture OF.

Inputs:
  input  peaks in Hz, or ratios when `source` is `tuning`
  amps   optional — the amplitudes beside those peaks, as `Peaks` emits them

Outputs:
  rgb        one row per degree, red green blue in 0 to 1. Wire this to an `image` viewer
  lightness  per degree, 0 to 1
  chroma     per degree, how saturated — 0 is grey
  hue        per degree, in degrees around the circle
  hex        the same palette as text, for a viewer or an agent to read

`method` is the choice worth making. `anchored` fixes the hue to the signal's own fingerprint, so
one signal keeps its colour as it drifts; `spectral` reads frequency as wavelength, the literal
reading; `tonotopic` spaces by how the ear places pitch; `consonance`, `harmonic` and `tenney`
colour by how simple each interval is, each by a different measure of simplicity; `mds` places the
degrees by their mutual distances; `derived` reads the colourspace off the signal itself.

`calibration` is the one to leave alone. It maps a descriptor onto a percentile so unrelated
signals do not collide, and the fitted ones ship as data files that this install does NOT carry —
so `none` is the honest default here, and a fitted one needs `build_calibration` over your own
extractor first.
"""

import warnings

import numpy as np
from biotuner.biocolors import palette_from_signal, palette_from_tuning
import goofi

METHODS = ["anchored", "spectral", "tonotopic", "consonance", "harmonic", "tenney", "mds", "derived"]


class BioColors(goofi.Node):
    """A palette from a signal's peaks or from a tuning."""

    TAGS = ["transform", "image"]
    INPUTS = {
        "input": goofi.InputSlot(goofi.DataType.ARRAY, required=True),
        "amps": goofi.InputSlot(goofi.DataType.ARRAY, required=False),
    }
    OUTPUTS = {
        "rgb": goofi.DataType.ARRAY,
        "lightness": goofi.DataType.ARRAY,
        "chroma": goofi.DataType.ARRAY,
        "hue": goofi.DataType.ARRAY,
        "hex": goofi.DataType.STRING,
    }
    PARAMS = {
        "color": {
            "source": goofi.StringParam(
                "peaks", ["peaks", "tuning"], doc="Whether the input is frequencies in Hz or ratios."
            ),
            "method": goofi.StringParam("anchored", METHODS, doc="What the hue is a picture of."),
            "fund": goofi.FloatParam(1.0, 0.001, 1000.0, doc="tuning: the frequency ratio 1 stands for."),
            "calibration": goofi.StringParam(
                "none", ["none", "tuning_v1", "eeg_sleep_v1"], doc="A fitted calibration, which this install may not carry."
            ),
        }
    }

    def process(self, input, amps=None):
        p = self.params.color
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("BioColors reads a list of peaks or ratios, not a single number")
        given = None if amps is None else np.asarray(amps.data, dtype=np.float64)

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        arows = None if given is None else given.reshape(-1, given.shape[-1])
        made = [self._palette(row, None if arows is None else arows[i % arows.shape[0]], p) for i, row in enumerate(rows)]

        wide = max((c.shape[0] for c, _ in made), default=0) or 1
        rgb = np.full((rows.shape[0], wide, 3), np.nan)
        lch = np.full((rows.shape[0], 3, wide), np.nan)
        words = []
        for i, (colours, spec) in enumerate(made):
            rgb[i, : colours.shape[0]] = colours
            lch[i, :, : spec.shape[1]] = spec
            words.append(" ".join("#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c) for c in colours))

        return {
            "rgb": rgb.reshape(lead + (wide, 3)).astype(np.float32),
            "lightness": lch[:, 0].reshape(lead + (wide,)).astype(np.float32),
            "chroma": lch[:, 1].reshape(lead + (wide,)).astype(np.float32),
            "hue": lch[:, 2].reshape(lead + (wide,)).astype(np.float32),
            "hex": "; ".join(w for w in words if w),
        }

    def _palette(self, row, row_amps, p):
        degrees = np.asarray([v for v in row if np.isfinite(v) and v > 0], dtype=np.float64)
        # A palette needs something to colour, and the mappings that space degrees need two.
        if degrees.size < 2:
            return np.zeros((1, 3)), np.zeros((3, 1))
        amp = None
        if row_amps is not None:
            amp = np.asarray([v for v in row_amps if np.isfinite(v)], dtype=np.float64)[: degrees.size]
            amp = None if amp.size < degrees.size else amp

        # The calibration warns on every call when the fitted percentiles do not cover a signal,
        # which at frame rate is a stderr flood rather than news.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if p.source == "tuning":
                pal = palette_from_tuning(degrees.tolist(), fund=p.fund, method=p.method, calibration=p.calibration)
            else:
                pal = palette_from_signal(
                    degrees.tolist(), amps=None if amp is None else amp.tolist(), method=p.method, calibration=p.calibration
                )

        colours = np.clip(np.asarray(pal.rgb, dtype=np.float64), 0.0, 1.0)
        spec = np.vstack([np.asarray(pal.spec.L), np.asarray(pal.spec.C), np.asarray(pal.spec.h)])
        return colours, spec
