import numpy as np
import pytest

from goofi.data import DTYPE_TO_TYPE, Data, DataType

from .utils import list_data_types


@pytest.mark.parametrize("dtype", list_data_types())
def test_create_data(dtype):
    # all dtype checks should pass
    Data(dtype, dtype.empty(), {})

    # data is None, should raise ValueError
    with pytest.raises(ValueError):
        Data(dtype, None, {})

    # metadata is None, should raise ValueError
    with pytest.raises(ValueError):
        Data(dtype, dtype.empty(), None)

    # dtype is None, should raise ValueError
    with pytest.raises(ValueError):
        Data(None, dtype.empty(), {})

    # make sure all other dtypes raise a ValueError
    for other_dtype in list_data_types():
        if other_dtype == dtype:
            continue
        with pytest.raises(ValueError):
            Data(dtype, other_dtype.empty(), {})


@pytest.mark.parametrize("dtype", list_data_types())
def test_dtype_map(dtype):
    assert dtype in DTYPE_TO_TYPE, f"Missing entry in DTYPE_TO_TYPE for dtype {dtype}."


def test_meta_dtype_is_set_for_every_data_type():
    # ARRAY → the numpy element dtype; STRING/TABLE → a content label. Auto-set
    # next to `shape`, so the metadata inspector always has it without recomputing.
    arr = Data(DataType.ARRAY, np.zeros(3, dtype=np.float32), {})
    assert arr.meta["dtype"] == "float32"
    assert arr.meta["shape"] == (3,)

    u8 = Data(DataType.ARRAY, np.zeros((2, 2), dtype=np.uint8), {})
    assert u8.meta["dtype"] == "uint8"

    assert Data(DataType.STRING, "hello", {}).meta["dtype"] == "str"
    assert Data(DataType.TABLE, {}, {}).meta["dtype"] == "table"


def test_array_never_carries_a_float_wider_than_float32():
    # float64 (the numpy default for most ops) is downcast at the Data boundary —
    # the single chokepoint — so nothing in the system ever carries float64.
    d = Data(DataType.ARRAY, np.array([1.0, 2.0, 3.0], dtype=np.float64), {})
    assert d.data.dtype == np.float32
    assert d.meta["dtype"] == "float32"
    # float128 (if the platform has it) is also narrowed to float32.
    if hasattr(np, "float128"):
        assert Data(DataType.ARRAY, np.zeros(2, dtype=np.float128), {}).data.dtype == np.float32


def test_array_preserves_float32_float16_and_integer_dtypes():
    # Only floats WIDER than 32-bit are touched — float32/float16/int stay as-is.
    assert Data(DataType.ARRAY, np.zeros(2, dtype=np.float32), {}).data.dtype == np.float32
    assert Data(DataType.ARRAY, np.zeros(2, dtype=np.float16), {}).data.dtype == np.float16
    assert Data(DataType.ARRAY, np.zeros(2, dtype=np.int64), {}).data.dtype == np.int64
    assert Data(DataType.ARRAY, np.zeros(2, dtype=np.uint8), {}).data.dtype == np.uint8


def test_construction_does_not_mutate_caller_meta():
    # Data is a value object: constructing it must NOT stamp shape/dtype/channels
    # into the caller's dict. Nodes routinely pass their own persistent `self.meta`
    # as the output meta; the old in-place mutation corrupted that shared dict every
    # tick (the systemic meta-mutation hazard). The Data owns an independent copy.
    meta = {"sfreq": 256.0}
    d = Data(DataType.ARRAY, np.ones(4, dtype=np.float32), meta)
    assert meta == {"sfreq": 256.0}  # caller's dict untouched
    assert d.meta is not meta
    assert d.meta["shape"] == (4,)
    assert d.meta["dtype"] == "float32"
    assert d.meta["channels"] == {}
    assert d.meta["sfreq"] == 256.0  # carried through


def test_construction_does_not_add_channels_to_caller_meta():
    # The absent-channels default ("channels": {}) must land on the Data's copy, not
    # the caller's dict — and a caller-supplied channels dict rides through intact.
    meta_no_chan = {"x": 1}
    Data(DataType.ARRAY, np.ones(2, dtype=np.float32), meta_no_chan)
    assert "channels" not in meta_no_chan and "shape" not in meta_no_chan

    chans = {"dim0": ["a", "b", "c", "d"]}
    d = Data(DataType.ARRAY, np.ones(4, dtype=np.float32), {"channels": chans})
    assert d.meta["channels"]["dim0"] == ["a", "b", "c", "d"]


def test_string_and_table_construction_do_not_mutate_caller_meta():
    sm = {"k": "v"}
    Data(DataType.STRING, "hi", sm)
    assert sm == {"k": "v"}  # no "dtype" stamped onto caller
    tm = {"k": "v"}
    Data(DataType.TABLE, {}, tm)
    assert tm == {"k": "v"}
