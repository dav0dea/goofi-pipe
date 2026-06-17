"""Unit tests for the data-plane per-slot multiplexer."""
from goofi.bridge.data import _SlotMux


class _FakeFwd:
    def __init__(self):
        self.frames = []

    def push_threadsafe(self, frame: bytes) -> None:
        self.frames.append(frame)


def test_dispatch_fans_out_to_all_forwarders():
    mux = _SlotMux(ref=None, slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)
    mux.dispatch(b"frame")
    assert a.frames == [b"frame"]
    assert b.frames == [b"frame"]


def test_remove_keeps_others_and_reports_empty():
    mux = _SlotMux(ref=None, slot="out")
    a, b = _FakeFwd(), _FakeFwd()
    mux.add(a)
    mux.add(b)
    assert mux.remove(a) is False  # b still connected
    mux.dispatch(b"y")
    assert a.frames == []  # removed → no frames
    assert b.frames == [b"y"]
    assert mux.remove(b) is True  # last one out → empty
