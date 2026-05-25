"""Binary codec for goofi `Data` and `Message` objects.

Two layers:

- `encode_data` / `decode_data` produce a self-describing byte frame for the
  data plane (numpy arrays + metadata). The frame is prefixed with a `GOOF`
  magic header so accidental cross-talk on a misnamed channel fails loudly.
- `encode_message` / `decode_message` produce msgpack bytes for the control
  and status planes (no nested `Data` objects in those messages after the
  iceoryx2 refactor, but the codec still tolerates them via an ExtType hook
  so nodes can publish state that contains arrays/tables if needed).

Inside msgpack streams:

- numpy arrays travel as `ExtType(_EXT_NDARRAY, ...)` with a compact custom
  header so we avoid pickle and stay zero-allocation on the hot path.
- nested `Data` objects travel as `ExtType(_EXT_DATA, encode_data(...))`.
- numpy scalars are demoted to Python scalars via the default hook.
- tuples are demoted to lists.
"""
from __future__ import annotations

import struct
from typing import Any

import msgpack
import numpy as np

from goofi.data import Data, DataType

_MAGIC = b"GOOF"
_EXT_DATA = 0
_EXT_NDARRAY = 1

_NDARRAY_HEADER = struct.Struct("<BB")


def encode_data(d: Data) -> bytes:
    """Serialize a `Data` object to a `GOOF`-prefixed byte frame."""
    inner = msgpack.packb(
        {"dtype": d.dtype.value, "meta": d.meta, "body": d.data},
        default=_packb_default,
        use_bin_type=True,
    )
    return _MAGIC + inner


def decode_data(buf: bytes) -> Data:
    """Deserialize a `GOOF`-prefixed byte frame into a fresh `Data` object."""
    if buf[:4] != _MAGIC:
        raise ValueError(f"Invalid Data frame: bad magic {buf[:4]!r}")
    inner = msgpack.unpackb(buf[4:], ext_hook=_ext_hook, raw=False)
    dtype = DataType(inner["dtype"])
    body = inner["body"]
    meta = inner["meta"] or {}
    if dtype == DataType.TABLE and isinstance(body, dict):
        # ext_hook already turned nested ExtType payloads into Data objects.
        pass
    return Data(dtype=dtype, data=body, meta=meta)


def encode_message(m) -> bytes:
    """Serialize a `Message` to msgpack bytes."""
    return msgpack.packb(
        {"type": m.type.value, "content": m.content},
        default=_packb_default,
        use_bin_type=True,
    )


def decode_message(buf: bytes):
    """Deserialize msgpack bytes into a `Message`."""
    # Local import: message.py imports nothing heavy now, but keeps the
    # codec module importable in contexts that mock Message out.
    from goofi.message import Message, MessageType

    obj = msgpack.unpackb(buf, ext_hook=_ext_hook, raw=False)
    return Message(type=MessageType(obj["type"]), content=obj["content"])


def _packb_default(obj: Any):
    if isinstance(obj, Data):
        return msgpack.ExtType(_EXT_DATA, encode_data(obj))
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(_EXT_NDARRAY, _pack_ndarray(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot pack object of type {type(obj).__name__}")


def _ext_hook(code: int, data: bytes):
    if code == _EXT_DATA:
        return decode_data(data)
    if code == _EXT_NDARRAY:
        return _unpack_ndarray(data)
    return msgpack.ExtType(code, data)


def _pack_ndarray(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a)
    dtype_str = a.dtype.str.encode("ascii")
    return b"".join(
        [
            _NDARRAY_HEADER.pack(len(dtype_str), a.ndim),
            dtype_str,
            struct.pack(f"<{a.ndim}I", *a.shape),
            a.tobytes(),
        ]
    )


def _unpack_ndarray(buf: bytes) -> np.ndarray:
    dtype_len, ndim = _NDARRAY_HEADER.unpack_from(buf, 0)
    off = _NDARRAY_HEADER.size
    dtype_str = buf[off : off + dtype_len].decode("ascii")
    off += dtype_len
    shape = struct.unpack_from(f"<{ndim}I", buf, off)
    off += 4 * ndim
    return np.frombuffer(buf, dtype=np.dtype(dtype_str), offset=off).reshape(shape).copy()
