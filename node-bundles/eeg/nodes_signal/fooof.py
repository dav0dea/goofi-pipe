"""Fooof — splits a spectrum into its aperiodic 1/f part and its peaks, with specparam.

Takes a `Psd` frame: `[.., F]` with the frequencies on its last axis. `aperiodic` is the offset
and exponent per spectrum (plus the knee in `knee` mode); `peaks` is `[.., max_peaks, 3]` of
centre frequency, power over the aperiodic fit and bandwidth, NaN past the peaks found. A
spectrum the model cannot fit answers NaN rather than stopping the stream.
"""

import warnings

import numpy as np
from specparam import SpectralModel
import goofi


class Fooof(goofi.Node):
    """FOOOF: the aperiodic 1/f slope and the peaks that rise above it."""

    INPUTS = {"psd": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"aperiodic": goofi.DataType.ARRAY, "peaks": goofi.DataType.ARRAY}
    PARAMS = {
        "fooof": {
            "mode": goofi.StringParam(
                "fixed", ["fixed", "knee"], doc="`knee` adds a bend where the 1/f slope flattens at low frequency."
            ),
            "max_peaks": goofi.IntParam(6, 1, 20, doc="Peaks to fit at most, and the width of `peaks`."),
            "peak_width_min": goofi.FloatParam(0.5, 0.0, 100.0, doc="Narrowest peak fitted, in Hz."),
            "peak_width_max": goofi.FloatParam(12.0, 0.0, 512.0, doc="Widest peak fitted, in Hz."),
            "freq_min": goofi.FloatParam(0.0, 0.0, 1000.0, doc="Lowest frequency fitted; 0 takes the spectrum's own edge."),
            "freq_max": goofi.FloatParam(0.0, 0.0, 1000.0, doc="Highest frequency fitted; 0 takes the spectrum's own edge."),
        }
    }

    def process(self, psd):
        p = self.params.fooof
        x = np.asarray(psd.data, dtype=np.float64)
        last = f"dim{x.ndim - 1}"
        freqs = np.asarray(psd.meta["channels"][last], dtype=np.float64)
        rows = x.reshape(-1, x.shape[-1])
        # The DC bin has no place on a log-frequency axis, and specparam says so on every fit.
        freq_range = (p.freq_min or freqs[freqs > 0].min(), p.freq_max or freqs.max())
        model = SpectralModel(
            aperiodic_mode=p.mode,
            peak_width_limits=(p.peak_width_min, p.peak_width_max),
            max_n_peaks=p.max_peaks,
            verbose=False,
        )
        ap_names = ["offset", "knee", "exponent"] if p.mode == "knee" else ["offset", "exponent"]
        aperiodic = np.full((rows.shape[0], len(ap_names)), np.nan)
        peaks = np.full((rows.shape[0], p.max_peaks, 3), np.nan)
        for i, row in enumerate(rows):
            # A fit that overflows on the way to its answer, or fails, is a NaN row and not a log line
            # — and specparam reports its own failed fit as `has_model` unset, never as a raise.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(freqs, row, freq_range=freq_range)
                except Exception:
                    continue
            if not model.results.has_model:
                continue
            aperiodic[i] = model.get_params("aperiodic")
            found = np.atleast_2d(model.get_params("peak"))
            if found.size:
                peaks[i, : len(found)] = found

        lead = x.shape[:-1]
        axes = {k: v for k, v in psd.meta.get("channels", {}).items() if k != last}
        # A knee the spectrum does not have fits as a number float32 cannot hold; inf is the answer.
        with np.errstate(over="ignore"):
            aperiodic = aperiodic.astype(np.float32)
        return {
            "aperiodic": (
                aperiodic.reshape(lead + (len(ap_names),)),
                {"channels": {**axes, last: ap_names}},
            ),
            "peaks": (
                peaks.reshape(lead + (p.max_peaks, 3)).astype(np.float32),
                {"channels": {**axes, f"dim{x.ndim}": ["cf", "pw", "bw"]}},
            ),
        }
