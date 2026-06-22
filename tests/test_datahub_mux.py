"""Unit tests for the data-plane per-slot multiplexer (viewer-adapters).

The mux owns ONE iceoryx2 subscription per (uid, slot); `dispatch` now receives a
decoded float `Data` and, for each distinct viewer kind among the connected
forwarders, adapts+encodes the frame exactly once before fanning it out — so two
viewers of different kinds on one slot each get their own representation, and N
viewers of the same kind share a single adapt/encode.
"""
import numpy as np

import goofi.bridge.data as data_mod
from goofi.bridge.data import _SlotMux
from goofi.codec import decode_data
from goofi.data import Data, DataType


class _FakeFwd:
    def __init__(self, kind: str):
        self.kind = kind
        self.frames = []

    def push_threadsafe(self, frame: bytes) -> None:
        self.frames.append(frame)


def _rgb():
    return Data(DataType.ARRAY, np.zeros((2, 2, 3), dtype=np.float32), {})


def _line():
    return Data(DataType.ARRAY, np.linspace(0, 1, 10, dtype=np.float32), {})


def test_dispatch_adapts_per_kind():
    mux = _SlotMux(ref=None, slot="out")
    img, line = _FakeFwd("image"), _FakeFwd("line")
    mux.add(img)
    mux.add(line)

    mux.dispatch(_rgb())  # one decoded float Data fans out as two representations

    assert decode_data(img.frames[-1]).data.dtype == np.uint8
    assert decode_data(line.frames[-1]).data.dtype == np.float16


def test_same_kind_forwarders_share_one_adapt(monkeypatch):
    calls = {"n": 0}
    real = data_mod.adapt

    def counting(d, kind):
        calls["n"] += 1
        return real(d, kind)

    monkeypatch.setattr(data_mod, "adapt", counting)
    mux = _SlotMux(ref=None, slot="out")
    a, b = _FakeFwd("line"), _FakeFwd("line")
    mux.add(a)
    mux.add(b)

    mux.dispatch(_line())

    assert calls["n"] == 1  # one adapt/encode for two same-kind forwarders
    assert len(a.frames) == 1 and len(b.frames) == 1


def test_remove_keeps_others_and_reports_empty():
    mux = _SlotMux(ref=None, slot="out")
    a, b = _FakeFwd("line"), _FakeFwd("line")
    mux.add(a)
    mux.add(b)
    assert mux.remove(a) is False  # b still connected
    mux.dispatch(_line())
    assert a.frames == []  # removed → no frames
    assert len(b.frames) == 1
    assert mux.remove(b) is True  # last one out → empty
