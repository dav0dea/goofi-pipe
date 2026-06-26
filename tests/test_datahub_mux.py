"""Unit tests for the data-plane per-slot multiplexer (Option C relay).

The mux owns ONE NodeRef view subscription per (uid, slot). The node has already
reduced + encoded the frame, so `dispatch` fans the node-reduced bytes to every
connected forwarder VERBATIM — no manager-side decode/adapt/re-encode. The mux
also folds the connected forwarders' ViewSpecs and pushes the fold to the node.
"""
from goofi.bridge.data import _SlotMux


class _FakeFwd:
    def __init__(self, spec=None):
        self.spec = spec or {"axes": [], "version": 0}
        self.frames = []

    def push_threadsafe(self, buf: bytes) -> None:
        self.frames.append(buf)


class _FakeRef:
    def __init__(self):
        self.specs = []

    def set_viewspec(self, slot, spec):
        self.specs.append((slot, spec))


def test_dispatch_fans_raw_bytes_verbatim():
    mux = _SlotMux(ref=_FakeRef(), slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)

    mux.dispatch(b"REDUCED")

    assert a.frames == [b"REDUCED"]
    assert b.frames == [b"REDUCED"]
    assert a.frames[0] is b.frames[0]  # same bytes fanned out, never re-encoded


def test_remove_keeps_others_and_reports_empty():
    mux = _SlotMux(ref=_FakeRef(), slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)
    assert mux.remove(a) is False  # b still connected
    mux.dispatch(b"x")
    assert a.frames == []  # removed → no frames
    assert b.frames == [b"x"]
    assert mux.remove(b) is True  # last one out → empty


def test_push_spec_if_changed_folds_richest_and_dedups():
    ref = _FakeRef()
    mux = _SlotMux(ref=ref, slot="out")
    a = _FakeFwd({"axes": [{"axis": -1, "max": 800, "method": "envelope"}], "version": 1})
    b = _FakeFwd({"axes": [{"axis": -1, "max": 2000, "method": "envelope"}], "version": 2})
    mux.add(a)
    mux.add(b)

    mux.push_spec_if_changed()
    assert len(ref.specs) == 1
    slot, folded = ref.specs[0]
    assert slot == "out"
    assert folded["axes"] == [{"axis": -1, "max": 2000, "method": "envelope"}]  # richest

    mux.push_spec_if_changed()  # unchanged fold → no extra push
    assert len(ref.specs) == 1

    mux.remove(b)  # drop the richer viewer → fold narrows → push
    mux.push_spec_if_changed()
    assert len(ref.specs) == 2
    assert ref.specs[-1][1]["axes"][0]["max"] == 800
