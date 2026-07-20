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
