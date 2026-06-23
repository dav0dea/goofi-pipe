"""The viewer adapters must keep `meta["dtype"]` reporting the NODE's true dtype,
not the wire dtype they downcast to — so the metadata inspector (which subscribes
through an adapter) stays dtype-accurate, exactly like the `__view__` stats."""
import numpy as np

from goofi.bridge.adapters import adapt
from goofi.data import Data, DataType


def test_line_adapter_preserves_source_dtype_not_float16():
    src = Data(DataType.ARRAY, np.zeros((4, 100), dtype=np.float32), {})
    assert src.meta["dtype"] == "float32"
    out = adapt(src, "line")
    assert out.data.dtype == np.float16  # body IS downcast for the wire
    assert out.meta["dtype"] == "float32"  # but the reported dtype stays true


def test_image_adapter_preserves_source_dtype_not_uint8():
    src = Data(DataType.ARRAY, np.zeros((8, 8, 3), dtype=np.float32), {})
    out = adapt(src, "image")
    assert out.data.dtype == np.uint8
    assert out.meta["dtype"] == "float32"


def test_passthrough_adapter_keeps_dtype():
    src = Data(DataType.STRING, "hi", {})
    assert adapt(src, "string").meta["dtype"] == "str"
