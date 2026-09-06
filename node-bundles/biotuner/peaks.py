"""Peaks — the frequencies a signal is loudest at, and how loud, from biotuner.

Takes a RAW TIME SERIES, not a spectrum: the transform happens inside, which is what `precision`
sets the resolution of. The last axis is time and is consumed; every axis before it survives, so
`[C, T]` in gives `[C, n_peaks]` out and one channel stays one row.

A peak is a frequency in Hz; an amplitude is that bin's power in dB, so it is normally NEGATIVE
and larger means louder. Both outputs are always `n_peaks` wide and padded with NaN where the
signal offered fewer — a band with nothing in it is a hole, never a shorter row, so the shape a
downstream node sees never depends on the data. A window with NO peak at all is that same hole:
the row comes back all NaN rather than faulting the node, which `harmonic_recurrence` on a short
or flat window would otherwise do constantly.

Extraction is the expensive half of biotuner, so it stands alone: one Peaks feeds Tuning,
Harmonicity and PeaksExtension without paying for the spectrum three times.

Inputs:
  input  a RAW time series. The last axis is time, and `sfreq` must be in the frame's metadata

Outputs:
  peaks  the loudest frequencies, in Hz, `n_peaks` wide and NaN-padded
  amps   each peak's power in dB — normally NEGATIVE, and larger means louder
"""

import numpy as np
from biotuner.biotuner_object import compute_biotuner
import goofi


class Peaks(goofi.Node):
    """The dominant spectral peaks of a signal, and their amplitudes."""

    TAGS = ["analysis"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"peaks": goofi.DataType.ARRAY, "amps": goofi.DataType.ARRAY}
    PARAMS = {
        "peaks": {
            "n_peaks": goofi.IntParam(5, 1, 10, doc="Peaks to look for, and the width of both outputs."),
            "f_min": goofi.FloatParam(2.0, 0.1, 50.0, doc="Lowest frequency a peak may sit at, in Hz."),
            "f_max": goofi.FloatParam(30.0, 1.0, 100.0, doc="Highest frequency a peak may sit at, in Hz."),
            "precision": goofi.FloatParam(0.5, 0.01, 10.0, doc="Resolution of the search, in Hz. Finer costs more."),
            "method": goofi.StringParam(
                "fixed",
                ["fixed", "EMD", "harmonic_recurrence", "EIMC"],
                doc="`fixed` takes the tallest bins. `EMD` decomposes the signal first and costs the most. "
                "`harmonic_recurrence` keeps peaks that are harmonics of one another; `EIMC` extends that.",
            ),
        }
    }

    def process(self, input):
        p = self.params.peaks
        x = np.asarray(input.data, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Peaks reads a time series, not a single number")
        # A signal with no sampling rate has no frequencies to name; biotuner would read the axis
        # as seconds and answer confidently in the wrong units, so refuse instead.
        sfreq = input.meta.get("sfreq")
        if not sfreq:
            raise ValueError("Peaks needs `sfreq` in the frame's metadata")

        lead, rows = x.shape[:-1], x.reshape(-1, x.shape[-1])
        out_p = np.full((rows.shape[0], p.n_peaks), np.nan)
        out_a = np.full((rows.shape[0], p.n_peaks), np.nan)
        for i, row in enumerate(rows):
            bt = compute_biotuner(float(sfreq), peaks_function=p.method, precision=p.precision)
            try:
                bt.peaks_extraction(row, min_freq=p.f_min, max_freq=p.f_max, n_peaks=p.n_peaks)
            except ValueError as e:
                # A band with nothing in it is a HOLE, not a fault: biotuner raises where it finds
                # no peak, and its own group code substitutes empty arrays for exactly this text.
                # `harmonic_recurrence` on a short or flat window says it often. Anything else is
                # a real failure and goes up.
                if "No peak detected" not in str(e):
                    raise
                continue
            pk = np.asarray(bt.peaks, dtype=np.float64).ravel()[: p.n_peaks]
            am = np.asarray(bt.amps, dtype=np.float64).ravel()[: p.n_peaks]
            out_p[i, : pk.size] = pk
            out_a[i, : am.size] = am

        shape = lead + (p.n_peaks,)
        meta = {"sfreq": sfreq}
        return {
            "peaks": (out_p.reshape(shape).astype(np.float32), meta),
            "amps": (out_a.reshape(shape).astype(np.float32), meta),
        }
