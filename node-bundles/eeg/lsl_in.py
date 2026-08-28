"""LSLIn — receives a Lab Streaming Layer stream, as `[channels, samples]` per tick.

The stream is resolved in the background and connected the moment it appears, so the node may
be added before the device is on: it stays silent until then, and it reconnects on its own if
the stream drops. Channel labels and the sample rate come from the stream's own description.
"""

import numpy as np
import pylsl
import goofi


class LSLIn(goofi.Node):
    """Receive an LSL stream: [channels, samples] per tick, with its labels and rate."""

    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PRODUCER = True
    PARAMS = {
        "lsl": {
            "name": goofi.StringParam(
                "", options=[""], refresh=True, doc="The stream's name; empty takes any stream of `type`."
            ),
            "type": goofi.StringParam("EEG", doc="The stream's type; empty takes any type."),
        }
    }

    def setup(self):
        self.wanted = None
        self.resolver = None
        self.inlet = None

    def process(self):
        p = self.params.lsl
        if (p.name, p.type) != self.wanted:
            self.wanted = (p.name, p.type)
            self.inlet = None
            pred = " and ".join(f"{k}='{v}'" for k, v in (("name", p.name), ("type", p.type)) if v)
            self.resolver = pylsl.ContinuousResolver(pred=pred or None)
        if self.inlet is None:
            found = self.resolver.results()
            if not found:
                return None
            self.inlet = pylsl.StreamInlet(found[0], recover=True)
            info = self.inlet.info()
            self.sfreq = info.nominal_srate() or None
            self.channels = labels(info)
        samples, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=32768)
        if not samples:
            return None
        return np.asarray(samples, dtype=np.float32).T, {"sfreq": self.sfreq, "channels": {"dim0": self.channels}}

    def refresh_lsl_name(self):
        return [""] + sorted({s.name() for s in pylsl.resolve_streams(wait_time=1.0)})


def labels(info):
    ch = info.desc().child("channels").child("channel")
    out = []
    for k in range(info.channel_count()):
        out.append(ch.child_value("label") or f"{info.type()} {k + 1}")
        ch = ch.next_sibling()
    return out
