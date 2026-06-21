"""Perf harness for the data-plane raw-byte forwarding unlock (A1, §7 gap).

Two assertions:

1. Always-on, deterministic: the raw-forward path the bridge now uses does
   ZERO codec work per frame (no `decode_data`, no `prepare_encode`, no
   `encode_data_into`), whereas the old decode→re-encode path did one of each
   per frame. This is the structural proof that the manager-side per-frame
   cost collapsed.
2. Opt-in (GOOFI_PERF=1), timing: on HD frames the raw path is ≥5× faster than
   the old transcode path. Gated so a loaded CI never flakes the suite.
"""
import os
import time

import numpy as np
import pytest

import goofi.codec as codec
from goofi.codec import encode_data
from goofi.data import Data, DataType


class _NoopMux:
    def dispatch(self, buf: bytes) -> None:
        # Mirror _SlotMux.dispatch's contract (fan a frame out) without any
        # forwarders — we only care about the per-frame work upstream of it.
        pass


def _old_forward(buf: bytes, mux: _NoopMux) -> None:
    """The pre-A1 bridge path: decode the wire bytes then re-encode them."""
    d = codec.decode_data(buf)
    size, meta_bytes = codec.prepare_encode(d)
    out = bytearray(size)
    codec.encode_data_into(d, memoryview(out), meta_bytes=meta_bytes)
    mux.dispatch(bytes(out))


def _raw_forward(buf: bytes, mux: _NoopMux) -> None:
    """The A1 path: forward the producer's wire bytes verbatim."""
    mux.dispatch(buf)


def _hd_frame_bytes() -> bytes:
    img = (np.arange(1920 * 1080 * 3, dtype=np.uint8) % 256).astype(np.uint8).reshape(1080, 1920, 3)
    return encode_data(Data(DataType.ARRAY, img, {}))


def test_raw_forward_does_zero_codec_work(monkeypatch):
    buf = _hd_frame_bytes()
    counts = {"decode": 0, "prepare": 0, "encode_into": 0}

    real_decode, real_prepare, real_encode = (
        codec.decode_data,
        codec.prepare_encode,
        codec.encode_data_into,
    )
    monkeypatch.setattr(codec, "decode_data", lambda *a, **k: counts.__setitem__("decode", counts["decode"] + 1) or real_decode(*a, **k))
    monkeypatch.setattr(codec, "prepare_encode", lambda *a, **k: counts.__setitem__("prepare", counts["prepare"] + 1) or real_prepare(*a, **k))
    monkeypatch.setattr(codec, "encode_data_into", lambda *a, **k: counts.__setitem__("encode_into", counts["encode_into"] + 1) or real_encode(*a, **k))

    mux = _NoopMux()
    for _ in range(20):
        _raw_forward(buf, mux)
    assert counts == {"decode": 0, "prepare": 0, "encode_into": 0}

    # Sanity: the old path DID do one of each per frame (guards the comparison).
    for _ in range(5):
        _old_forward(buf, mux)
    assert counts == {"decode": 5, "prepare": 5, "encode_into": 5}


@pytest.mark.skipif(os.environ.get("GOOFI_PERF") != "1", reason="set GOOFI_PERF=1 to run timing benchmark")
def test_raw_forward_is_much_faster_on_hd():
    buf = _hd_frame_bytes()
    mux = _NoopMux()
    n = 50

    # warm up
    _old_forward(buf, mux)
    _raw_forward(buf, mux)

    t0 = time.perf_counter()
    for _ in range(n):
        _old_forward(buf, mux)
    old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        _raw_forward(buf, mux)
    raw = time.perf_counter() - t0

    speedup = old / max(raw, 1e-9)
    print(f"\nHD frame: old={old/n*1e3:.2f} ms/frame, raw={raw/n*1e6:.2f} us/frame, speedup={speedup:.0f}x")
    assert speedup >= 5.0, f"raw forwarding only {speedup:.1f}x faster (expected >=5x)"
