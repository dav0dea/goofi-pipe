"""Psd — Welch-style power spectral density over the last axis.

A stopgap Python implementation used to exercise the remote (subprocess) node tier + the
typed SHM transport; the real node will eventually be native Rust. Reads the sampling rate
from the worker-provided `goofi_meta['sfreq']` (so the PSD magnitude is physically scaled),
falling back to 1.0 when absent. Preserves channels: [C, T] -> [C, T//2 + 1].
"""

import numpy as np


def process(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    sfreq = goofi_meta.get("sfreq") or 1.0
    win = np.hanning(x.shape[-1])
    xf = np.fft.rfft(x * win, axis=-1)
    psd = (np.abs(xf) ** 2) / (sfreq * (win ** 2).sum())
    return psd.astype(np.float32)
