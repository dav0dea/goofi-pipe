import numpy as np
import goofi


def test_roundtrip_array_and_meta():
    d = goofi.Data(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), {"sfreq": 250.0})
    out = d.data
    assert out.dtype == np.float32
    assert out.shape == (1, 3)
    assert list(out.ravel()) == [1.0, 2.0, 3.0]
    assert d.meta["sfreq"] == 250.0


def test_non_f32_input_is_cast():
    d = goofi.Data(np.array([1, 2, 3], dtype=np.int32))
    assert d.data.dtype == np.float32
    assert list(d.data) == [1.0, 2.0, 3.0]


def test_channels_meta_roundtrips_as_nested_dict():
    d = goofi.Data(
        np.array([1.0, 2.0], dtype=np.float32),
        {"channels": {"dim0": ["Fz", "Cz"]}},
    )
    assert d.meta["channels"]["dim0"] == ["Fz", "Cz"]


def test_meta_value_types_roundtrip():
    # Pins the pythonized (untagged-serde) meta map: each scalar/list/nested type must
    # survive Python->Rust->Python with its type intact. This is the regression guard for
    # the `#[serde(untagged)]` MetaValue variant ordering + the Bytes/Axes skips — reorder
    # or unskip wrongly and one of these flips type (e.g. a list read back as bytes).
    meta = {
        "sfreq": 250.0,
        "index": 7,
        "flag": True,
        "note": "eeg",
        "freqs": [1.0, 2.5, 4.0],
        "ints": [1, 2, 3],
        "nested": {"a": 1, "b": [2.0, 3.0]},
    }
    m = goofi.Data(np.array([1.0], dtype=np.float32), meta).meta
    assert m["sfreq"] == 250.0 and isinstance(m["sfreq"], float)
    assert m["index"] == 7 and isinstance(m["index"], int)
    assert m["flag"] is True
    assert m["note"] == "eeg"
    assert m["freqs"] == [1.0, 2.5, 4.0]
    assert m["ints"] == [1, 2, 3]
    assert m["nested"] == {"a": 1, "b": [2.0, 3.0]}
