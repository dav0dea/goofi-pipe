"""EegPowerBands — the power in each EEG band, from a spectrum.

Takes a `Psd` frame: `[.., F]` with the frequencies on its last axis, and answers `[.., bands]`
with the band names on that axis. Power is the density integrated over the band, so it does not
move with the spectrum's resolution. A band is `lo-hi` in Hz; leave one empty to drop it.
"""

import numpy as np
import goofi

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


class EegPowerBands(goofi.Node):
    """Band power per channel: delta to gamma, absolute or as a share of the whole spectrum."""

    TAGS = ["analysis", "eeg"]
    INPUTS = {"psd": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"power": goofi.DataType.ARRAY}
    PARAMS = {
        "bands": {
            "delta": goofi.StringParam("1-4", doc="Hz, as `lo-hi`; empty drops the band."),
            "theta": goofi.StringParam("4-8"),
            "alpha": goofi.StringParam("8-13"),
            "beta": goofi.StringParam("13-30"),
            "gamma": goofi.StringParam("30-50"),
            "relative": goofi.BoolParam(False, doc="Divide each band by the power of the whole spectrum."),
        }
    }

    def process(self, psd):
        p = self.params.bands
        x = np.asarray(psd.data, dtype=np.float64)
        last = f"dim{x.ndim - 1}"
        freqs = np.asarray(psd.meta["channels"][last], dtype=np.float64)

        names, power = [], []
        for name in BANDS:
            spec = getattr(p, name).strip()
            if not spec:
                continue
            lo, hi = (float(v) for v in spec.split("-"))
            band = (freqs >= lo) & (freqs <= hi)
            names.append(name)
            power.append(np.trapezoid(x[..., band], freqs[band], axis=-1))
        power = np.stack(power, axis=-1)
        if p.relative:
            power = power / np.trapezoid(x, freqs, axis=-1)[..., None]

        axes = {k: v for k, v in psd.meta.get("channels", {}).items() if k != last}
        return power.astype(np.float32), {"channels": {**axes, last: names}}
