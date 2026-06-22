"""Data-plane per-frame cost contract for the viewer-adapters path (backlog #18, #3).

The bridge no longer forwards the producer's wire bytes verbatim (the A1 raw-
forward unlock was deliberately reversed by the viewer-adapters rework). Instead
the NodeRef pump decodes each slot's float ``Data`` ONCE, and ``_SlotMux.dispatch``
adapts + re-encodes it ONCE per distinct viewer kind, sharing that encoded frame
across every forwarder (viewer) of the same kind.

Two always-on, deterministic assertions characterise that contract:

1. ``dispatch`` adapts + encodes once per *distinct kind*, not once per forwarder —
   so the manager-side per-frame cost scales with the number of kinds on a slot,
   not the number of viewers attached.
2. The adapted representation is smaller on the wire than the source float frame
   (uint8 for images, float16 for line plots), and round-trips to that dtype.
"""
import numpy as np

from goofi.bridge import data as dataplane
from goofi.bridge.adapters import adapt
from goofi.bridge.data import _SlotMux
from goofi.codec import decode_data, encode_data
from goofi.data import Data, DataType


class _FakeFwd:
    """Stand-in for a connected WS forwarder: records the frames pushed to it."""

    def __init__(self, kind: str):
        self.kind = kind
        self.frames: list[bytes] = []

    def push_threadsafe(self, frame: bytes) -> None:
        self.frames.append(frame)


def _hd_image() -> Data:
    # float RGB HD frame — the pure-float Data a producer emits, pre-adapter.
    img = (np.arange(1080 * 1920 * 3, dtype=np.float32) % 256.0).reshape(1080, 1920, 3)
    return Data(DataType.ARRAY, img, {})


def test_dispatch_adapts_once_per_distinct_kind(monkeypatch):
    mux = _SlotMux(ref=None, slot="out")
    # Three image viewers + two line viewers on the SAME slot.
    fwds = [_FakeFwd("image"), _FakeFwd("image"), _FakeFwd("image"), _FakeFwd("line"), _FakeFwd("line")]
    for f in fwds:
        mux.add(f)

    counts = {"adapt": 0, "encode": 0}
    real_adapt, real_encode = dataplane.adapt, dataplane.encode_data
    monkeypatch.setattr(dataplane, "adapt", lambda d, k: counts.__setitem__("adapt", counts["adapt"] + 1) or real_adapt(d, k))
    monkeypatch.setattr(dataplane, "encode_data", lambda d: counts.__setitem__("encode", counts["encode"] + 1) or real_encode(d))

    mux.dispatch(_hd_image())

    # Two distinct kinds → exactly two adapt + two encode, NOT one per forwarder (5).
    assert counts == {"adapt": 2, "encode": 2}
    # Every forwarder received exactly one frame.
    assert all(len(f.frames) == 1 for f in fwds)
    # Same-kind forwarders share the identical encoded bytes (encoded once, fanned out).
    assert fwds[0].frames[0] is fwds[1].frames[0] is fwds[2].frames[0]
    assert fwds[3].frames[0] is fwds[4].frames[0]
    # Different kinds get different representations.
    assert fwds[0].frames[0] is not fwds[3].frames[0]


def test_adapted_frame_is_smaller_than_the_source_float_frame():
    img = _hd_image()
    raw = encode_data(img)  # what a verbatim forward (float32 HD frame) would have sent
    adapted = encode_data(adapt(img, "image"))
    assert len(adapted) < len(raw)  # uint8 body is ~4x smaller than float32
    assert decode_data(adapted).data.dtype == np.uint8

    sig = Data(DataType.ARRAY, np.arange(32 * 1000, dtype=np.float32).reshape(32, 1000), {})
    line = encode_data(adapt(sig, "line"))
    assert len(line) < len(encode_data(sig))  # float16 is half of float32
    assert decode_data(line).data.dtype == np.float16
