"""Data-plane per-frame cost contract for the Option C node-reduction relay.

The node reduces + encodes the frame ONCE on its reducer thread; the manager
subscribes once per (node, slot) and `_SlotMux.dispatch` fans the node-reduced
bytes to every forwarder VERBATIM. So the manager-side per-frame cost is O(viewers)
trivial pushes of the SAME bytes — no decode, no adapt, no re-encode — independent
of the number of viewers AND their kinds. The reduction (the bandwidth win) lives
in the node, before the cross-process copy.
"""
import numpy as np

from goofi.bridge.data import _SlotMux
from goofi.codec import encode_data
from goofi.data import Data, DataType
from goofi.node_reduce import reduce_for_view, viewspec_from_dict


class _FakeFwd:
    """Stand-in for a connected WS forwarder: records the frames pushed to it."""

    def __init__(self):
        self.frames: list[bytes] = []

    def push_threadsafe(self, buf: bytes) -> None:
        self.frames.append(buf)


def test_dispatch_fans_one_buffer_to_all_forwarders():
    mux = _SlotMux(ref=None, slot="out")
    fwds = [_FakeFwd() for _ in range(5)]
    for f in fwds:
        mux.add(f)

    mux.dispatch(b"NODE_REDUCED_FRAME")

    # Every forwarder received exactly one frame — the identical bytes object,
    # fanned out with no per-viewer / per-kind manager work.
    assert all(len(f.frames) == 1 for f in fwds)
    first = fwds[0].frames[0]
    assert all(f.frames[0] is first for f in fwds)


def test_node_reduced_frame_is_far_smaller_than_the_raw_float_frame():
    # HD image: a full-frame verbatim forward would have sent float32 1080x1920x3.
    img = Data(
        DataType.ARRAY,
        (np.arange(1080 * 1920 * 3, dtype=np.float32) % 256.0).reshape(1080, 1920, 3),
        {},
    )
    raw = encode_data(img)
    img_spec = viewspec_from_dict(
        {"axes": [{"axis": 0, "max": 720, "method": "area"},
                  {"axis": 1, "max": 1280, "method": "area"}]}
    )
    reduced = encode_data(reduce_for_view(img, img_spec))
    assert len(reduced) < len(raw) / 2  # 1080x1920 -> 720x1280 area downscale

    # 1 s of 32-channel 44.1 kHz audio -> ~2*2000 envelope per channel.
    sig = Data(DataType.ARRAY, np.arange(32 * 44100, dtype=np.float32).reshape(32, 44100), {})
    full = encode_data(sig)
    line_spec = viewspec_from_dict({"axes": [{"axis": -1, "max": 2000, "method": "envelope"}]})
    line = encode_data(reduce_for_view(sig, line_spec))
    assert len(line) < len(full) / 5
