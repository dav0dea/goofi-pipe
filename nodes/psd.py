"""Psd — Welch-style power spectral density over the last axis.

A stopgap Python implementation used to exercise the remote (subprocess) node tier over the
`goofi.serve` transport; the real node will eventually be native Rust. Authored to the one
`goofi.Node` class contract: declares its slots + a `welch.nperseg` param (proving param
discovery reaches the manifest), and reads the sampling rate from `data.meta["sfreq"]` (so the
PSD magnitude is physically scaled), falling back to 1.0 when absent. Preserves channels:
[C, T] -> [C, T//2 + 1].
"""

import numpy as np
import goofi


class Psd(goofi.Node):
    @staticmethod
    def config_input_slots():
        # `required`: process() reads `data.data` unconditionally, so a tick with an empty slot
        # cannot work. The engine refuses the tick before process() is entered.
        return {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}

    @staticmethod
    def config_output_slots():
        return {"psd": goofi.DataType.ARRAY}

    @staticmethod
    def config_params():
        return {
            "welch": {
                "nperseg": goofi.IntParam(
                    256, 16, 4096, doc="Window length in samples: longer windows resolve finer frequencies."
                )
            }
        }

    def process(self, data):
        x = np.asarray(data.data, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        sfreq = data.meta.get("sfreq") or 1.0
        win = np.hanning(x.shape[-1])
        xf = np.fft.rfft(x * win, axis=-1)
        psd = (np.abs(xf) ** 2) / (sfreq * (win ** 2).sum())
        # A length-changing node explicitly manages its output meta: the input's time-domain
        # sfreq is meaningless for the spectrum, so emit the frequency axis instead. (A bare
        # return would correctly drop the input meta on a shape change.)
        freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq).tolist()
        return {"psd": (psd.astype(np.float32), {"freqs": freqs})}
