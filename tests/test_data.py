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
