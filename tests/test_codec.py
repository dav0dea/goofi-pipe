"""Codec contract tests — the GOOF binary wire format (codec.py).

The format is hand-mirrored in frontend/src/lib/codec/decode.ts, so it has TWO
nets here:
  1. Round-trip identity for every dtype (encode -> decode -> equal).
  2. A committed golden fixture (tests/codec_golden.json) decoded + re-encoded to
     pin the exact bytes. The SAME fixture is decoded by decode.test.ts, so the
     py encoder and the ts decoder can never drift silently.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from goofi.codec import decode_data, encode_data
from goofi.data import Data, DataType

GOLDEN = json.loads((Path(__file__).parent / "codec_golden.json").read_text(encoding="utf-8"))


# ---- round-trip identity -----------------------------------------------------

def _assert_array_eq(a: Data, b: Data):
    assert a.dtype == b.dtype
    assert a.data.dtype == b.data.dtype
    assert a.data.shape == b.data.shape
    assert np.array_equal(a.data, b.data)


@pytest.mark.parametrize("arr", [
    np.array([1.0, 2.5, -3.0], dtype=np.float32),
    np.array([[1.0, 2.0], [0.5, -4.0]], dtype=np.float16),
    np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4),
    np.array([0, -5, 2147483648], dtype=np.int64),
    np.zeros(0, dtype=np.float32),  # empty
])
def test_array_round_trip(arr):
    d = Data(DataType.ARRAY, arr, {"sfreq": 256.0})
    rt = decode_data(encode_data(d))
    _assert_array_eq(d, rt)
    assert rt.meta["sfreq"] == 256.0


def test_string_round_trip():
    d = Data(DataType.STRING, "héllo · 世界", {})
    rt = decode_data(encode_data(d))
    assert rt.dtype == DataType.STRING and rt.data == "héllo · 世界"


def test_table_round_trip_nested():
    inner = Data(DataType.TABLE, {"x": Data(DataType.ARRAY, np.float32([1, 2]), {})}, {})
    d = Data(DataType.TABLE, {
        "sig": Data(DataType.ARRAY, np.float32([1, 2, 3]), {}),
        "label": Data(DataType.STRING, "ok", {}),
        "nested": inner,
    }, {})
    rt = decode_data(encode_data(d))
    assert set(rt.data) == {"sig", "label", "nested"}
    _assert_array_eq(d.data["sig"], rt.data["sig"])
    assert rt.data["label"].data == "ok"
    _assert_array_eq(d.data["nested"].data["x"], rt.data["nested"].data["x"])


def test_channels_meta_survives_round_trip():
    d = Data(DataType.ARRAY, np.float32([10, 20, 30, 40]),
             {"channels": {"dim0": ["Fz", "Cz", "Pz", "Oz"]}})
    rt = decode_data(encode_data(d))
    assert rt.meta["channels"]["dim0"] == ["Fz", "Cz", "Pz", "Oz"]


# ---- golden fixture: pins the exact bytes + cross-language contract ----------

def _norm(meta: dict) -> dict:
    # Python's Data re-stamps meta["shape"] as a numpy tuple on construction; the
    # wire (msgpack) carries it as a list, which is exactly what the TS decoder
    # surfaces. Normalize tuples->lists so the golden compares the WIRE view that
    # both languages must agree on.
    return json.loads(json.dumps(meta, default=list))


def _check_decoded(d: Data, exp: dict):
    assert d.dtype.name == exp["dtype"]
    assert _norm(d.meta) == exp["meta"]
    if exp["dtype"] == "ARRAY":
        assert d.data.dtype.str == exp["arrayDtype"]
        assert list(d.data.shape) == exp["shape"]
        assert d.data.reshape(-1).tolist() == exp["values"]
    elif exp["dtype"] == "STRING":
        assert d.data == exp["value"]
    else:  # TABLE
        assert set(d.data) == set(exp["entries"])
        for k, sub in exp["entries"].items():
            _check_decoded(d.data[k], sub)


@pytest.mark.parametrize("entry", GOLDEN["entries"], ids=lambda e: e["name"])
def test_golden_decodes_and_reencodes_stably(entry):
    buf = bytes.fromhex(entry["hex"])
    d = decode_data(buf)
    _check_decoded(d, entry["decoded"])
    # re-encoding the decoded Data must reproduce the exact committed bytes —
    # this is what pins the wire format against accidental drift.
    assert encode_data(d).hex() == entry["hex"]


def test_golden_version_is_current():
    assert GOLDEN["version"] == 2
