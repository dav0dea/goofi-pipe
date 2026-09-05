"""LslOut — publishes a frame as a Lab Streaming Layer stream.

`[C]` pushes one sample, `[C, T]` a chunk. The channel names and the sample rate come from the
frame, so the outlet describes the signal it carries; it is rebuilt when either of those changes,
because an outlet's description is fixed for its life.
"""

import numpy as np
import pylsl
import goofi


class LslOut(goofi.Node):
    """Publish a frame as an LSL stream, with the labels and rate it arrived with."""

    TAGS = ["output"]
    INPUTS = {"input": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    PARAMS = {
        "lsl": {
            "name": goofi.StringParam("goofi", doc="The name other programs resolve the stream by."),
            "type": goofi.StringParam("EEG", doc="The stream's type, such as EEG or Audio."),
        }
    }

    def setup(self):
        self.outlet = None
        self.built = None

    def process(self, input):
        p = self.params.lsl
        raw = np.asarray(input.data, dtype=np.float32)
        if raw.ndim not in (1, 2):
            raise ValueError(f"needs [C] or [C, T], got {list(raw.shape)}")
        x = raw[:, None] if raw.ndim == 1 else raw
        sfreq = float(input.meta.get("sfreq") or 0.0)
        labels = list(input.meta.get("channels", {}).get("dim0") or [])

        wanted = (p.name, p.type, x.shape[0], sfreq, tuple(labels))
        if wanted != self.built:
            info = pylsl.StreamInfo(p.name, p.type, x.shape[0], sfreq, pylsl.cf_float32, f"goofi-{p.name}")
            channels = info.desc().append_child("channels")
            for i in range(x.shape[0]):
                name = labels[i] if i < len(labels) else f"{p.type} {i + 1}"
                channels.append_child("channel").append_child_value("label", str(name))
            self.outlet = pylsl.StreamOutlet(info)
            self.built = wanted

        if raw.ndim == 1:
            self.outlet.push_sample(x[:, 0])
        else:
            self.outlet.push_chunk(x.T.tolist())
        return None
