import time

import numpy as np

from goofi.audio.continuity import INDEX_META_KEY
from goofi.audio.ring import AudioRing
from goofi.data import DataType
from goofi.node import InputSlot, Node
from goofi.params import BoolParam, FloatParam, IntParam

_N_INPUTS = 8
# Drop an input from the mix if it stops delivering for this long, so a single
# stalled source can't wedge the whole mix (min-fill would sit at zero forever).
_STALE_SECONDS = 0.5


class Mixer(Node):
    """
    This node averages (or sums) several independently-clocked audio streams into a single output. Each input is buffered in its own jitter ring; on every tick the node emits the largest block that ALL currently-active inputs can supply (their minimum fill), combined sample-for-sample. The per-input rings absorb the per-tick block-size jitter of independent SampleClock generators — several Oscillators that each emit a slightly different number of samples per update still combine sample-aligned. An input that goes silent for a short window drops out of the mix automatically.

    Inputs:
    - in0..in7: Audio arrays to combine. Each is a lossless queue slot; unconnected inputs are ignored.

    Outputs:
    - out: The combined audio (average of the active inputs by default, or their sum), carrying the inputs' sfreq.
    """

    def config_input_slots():
        # Lossless queue slots: every emitted frame is drained and buffered, so no
        # samples are dropped before alignment. Index-deduped on write.
        return {f"in{i}": InputSlot(DataType.ARRAY, queue=True) for i in range(_N_INPUTS)}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "mixer": {
                "normalize": BoolParam(True, doc="Average the active inputs (True) or sum them (False)."),
                "buffer_ms": IntParam(50, 5, 500, doc="Per-input jitter-ring depth (ms at 48 kHz)."),
            },
            # Free-run fast enough to drain the queues promptly; the rings, not the
            # tick rate, set alignment. (Paced so the loop can't busy-spin.)
            "common": {"autotrigger": True, "max_frequency": FloatParam(500.0, 0.0, 1000.0)},
        }

    def setup(self):
        self._names = [f"in{i}" for i in range(_N_INPUTS)]
        cap = max(int(48000 * self.params.mixer.buffer_ms.value / 1000) * 4, 1024)
        self._rings = {n: AudioRing(cap, 1) for n in self._names}
        self._last_index = {n: None for n in self._names}
        self._last_seen = {n: 0.0 for n in self._names}
        self._active = set()
        self._sfreq = None

    def process(self, **inputs):
        now = time.time()

        # 1. Ingest fresh frames (dedup by continuity index) into each input's ring.
        for name in self._names:
            data = inputs.get(name)
            if data is None:
                continue
            idx = data.meta.get(INDEX_META_KEY)
            if idx is not None and idx == self._last_index[name]:
                continue  # stale re-read of the same emit
            self._last_index[name] = idx
            self._rings[name].write(np.asarray(data.data, dtype=np.float32).reshape(-1, 1))
            self._active.add(name)
            self._last_seen[name] = now
            if self._sfreq is None:
                self._sfreq = data.meta.get("sfreq")

        # 2. Retire inputs that have gone silent (and fully drained) so one stalled
        #    source can't pin the min-fill at zero and wedge the mix.
        self._active = {
            n for n in self._active
            if self._rings[n].fill > 0 or now - self._last_seen[n] < _STALE_SECONDS
        }
        if not self._active:
            return None

        # 3. Emit the largest block ALL active inputs can supply, combined.
        n_frames = min(self._rings[n].fill for n in self._active)
        if n_frames == 0:
            return None
        acc = np.zeros((n_frames, 1), dtype=np.float32)
        buf = np.zeros((n_frames, 1), dtype=np.float32)
        for n in self._active:
            self._rings[n].read_into(buf)
            acc += buf
        if self.params.mixer.normalize.value:
            acc /= len(self._active)

        meta = {"sfreq": self._sfreq} if self._sfreq is not None else {}
        return {"out": (acc.reshape(-1), meta)}
