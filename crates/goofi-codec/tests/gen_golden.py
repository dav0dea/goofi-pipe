#!/usr/bin/env python3
"""Generate authoritative GOOF v2 golden frames from the REAL legacy Python codec.

Run with a Python env that has numpy + msgpack and can import the legacy `goofi`
package from `src/` (e.g. the repo `.venv` or the back-reference checkout's venv):

    PYTHONPATH=src <python> crates/goofi-codec/tests/gen_golden.py

Writes crates/goofi-codec/tests/fixtures/goof_golden.json — a name -> {frame: [u8...],
dtype_tag} map. The Rust codec test reconstructs the SAME cases natively, encodes them,
and asserts body bytes are byte-identical and meta is semantically equal. The case list
here is mirrored exactly in tests/goof_golden.rs; the two must agree by construction.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from goofi.data import Data, DataType  # noqa: E402
from goofi.codec import encode_data  # noqa: E402


def arr(a):
    return Data(DataType.ARRAY, a, {})


cases = {}
# Arrays are always float32 — foreign dtypes are cast to f32 at the ingest boundary,
# so the wire only ever carries <f4. Non-f32 cases are intentionally absent.
cases["f32_1d"] = arr(np.array([1.0, 2.0, 3.0], dtype=np.float32))
cases["scalar_0d"] = Data(DataType.ARRAY, np.float32(3.0), {})  # 0-d -> shape (1,)
cases["empty_array"] = arr(np.array([], dtype=np.float32))  # shape (0,), 0 body bytes
cases["with_sfreq_channels"] = Data(
    DataType.ARRAY,
    np.arange(6, dtype=np.float32).reshape(2, 3),
    {"sfreq": 250.0, "channels": {"dim0": ["Fz", "Cz"]}},
)
cases["with_index"] = Data(
    DataType.ARRAY, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), {"index": 42}
)
cases["string"] = Data(DataType.STRING, "hello world éà", {})
cases["string_empty"] = Data(DataType.STRING, "", {})
cases["table"] = Data(
    DataType.TABLE,
    {
        "a": arr(np.array([1.0, 2.0], dtype=np.float32)),
        "b": Data(DataType.STRING, "x", {}),
    },
    {},
)
cases["table_empty"] = Data(DataType.TABLE, {}, {})

out = {}
for name, d in cases.items():
    frame = encode_data(d)
    out[name] = {"frame": list(frame), "dtype_tag": d.dtype.value}

fixtures = os.path.join(HERE, "fixtures")
os.makedirs(fixtures, exist_ok=True)
path = os.path.join(fixtures, "goof_golden.json")
with open(path, "w") as f:
    json.dump(
        {"_numpy": np.__version__, "cases": out}, f, indent=1, sort_keys=True
    )
print(f"wrote {len(out)} cases -> {path}")
