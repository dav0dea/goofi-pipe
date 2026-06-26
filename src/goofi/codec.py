"""Binary codec for goofi `Data` and `Message` objects.

Wire format
-----------

`Data` is serialized as a fixed-header + meta + body frame::

    offset  size  field
    ------  ----  ----------------------------------------------
    0       4     magic "GOOF"
    4       1     version (=2)
    5       1     dtype tag = DataType.value (0=ARRAY, 1=STRING, 2=TABLE)
    6       4     meta_len   (LE u32)  — msgpack-encoded meta dict
    10      4     body_len   (LE u32)
    14      *     meta bytes (msgpack)
    14+meta *     body bytes (dtype-specific)

    meta_len is u32 (not u16): nodes legitimately carry large coordinate
    arrays in meta — e.g. PSD/FFT store the full frequency axis under
    `channels["dim<axis>"]`, which exceeds 64 KB at fine resolution.

    ARRAY body:
        [u8 ndim][u8 dtype_str_len][dtype_str][ndim × LE u32 shape][raw ndarray bytes]
    STRING body:
        [utf-8 bytes]
    TABLE body:
        [LE u32 n_entries][repeated: u16 key_len, key, u32 value_len, encoded Data]

The format is designed so byte size is computable without serializing the
body. `data_byte_size(d)` returns the exact payload size; callers can loan
a precisely-sized iceoryx2 slice (or `bytearray`) and `encode_data_into`
writes directly into it — for ARRAY data, the numpy array bytes are
copied straight from the heap buffer to the destination view via the
buffer protocol, a single memcpy with no Python `bytes` intermediate.

`Message` envelope is still vanilla msgpack — control-plane messages are
small and benefit nothing from a custom format.
"""
from __future__ import annotations

import struct
from typing import Tuple, Union

import msgpack
import numpy as np

from goofi.data import Data, DataType

_MAGIC = b"GOOF"
_VERSION = 2
# fixed header — see module docstring
_HEADER = struct.Struct("<4sBBII")  # magic, version, dtype, meta_len (u32), body_len (u32)
_HEADER_SIZE = _HEADER.size  # 14
# ARRAY body sub-header — ndim + dtype_str_len, followed by dtype_str and shape u32s
_ARRAY_SUBHEADER = struct.Struct("<BB")


# ---------------------------------------------------------------------------
# Size computation (no allocation, no body serialization)
# ---------------------------------------------------------------------------


def prepare_encode(d: Data) -> Tuple[int, bytes]:
    """Pre-compute encoded size + packed meta for a `Data` object.

    For callers that publish the same `Data` to multiple publishers
    (multi-consumer fan-out), this lets you pack the meta dict once
    and reuse the result for every per-publisher `encode_data_into`
    call — important because msgpack-packing a meta dict is non-trivial
    relative to small payload sizes.
    """
    meta_bytes = _pack_meta(d.meta)
    body_size = _body_size(d)
    return _HEADER_SIZE + len(meta_bytes) + body_size, meta_bytes


def data_byte_size(d: Data) -> int:
    """Return the exact byte size `encode_data_into` would write for `d`."""
    return prepare_encode(d)[0]


def _pack_meta(meta: dict) -> bytes:
    """Serialize the meta dict via msgpack. Tuples → lists; numpy scalars → Python."""
    return msgpack.packb(meta, default=_msgpack_default, use_bin_type=True)


def _msgpack_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot msgpack-encode meta field of type {type(obj).__name__}")


def _body_size(d: Data) -> int:
    if d.dtype == DataType.ARRAY:
        a = d.data
        dtype_str = a.dtype.str.encode("ascii")
        return _ARRAY_SUBHEADER.size + len(dtype_str) + 4 * a.ndim + a.nbytes
    if d.dtype == DataType.STRING:
        return len(d.data.encode("utf-8"))
    if d.dtype == DataType.TABLE:
        # u32 n_entries + for each: u16 key_len + key + u32 value_len + value bytes
        size = 4
        for key, value in d.data.items():
            size += 2 + len(key.encode("utf-8")) + 4 + data_byte_size(value)
        return size
    raise ValueError(f"Unsupported dtype: {d.dtype}")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_data_into(d: Data, view: memoryview, meta_bytes: bytes = None) -> int:
    """Write the encoding of `d` into `view`. Returns bytes written.

    The caller must ensure `len(view) >= data_byte_size(d)`. `view` must
    be a writable memoryview of a contiguous byte buffer (e.g., a
    `bytearray` or an iceoryx2 SHM slice exposed as a memoryview).

    For ARRAY data, the numpy buffer is copied directly into `view` via
    the buffer protocol — no intermediate Python `bytes` allocation.

    If `meta_bytes` is provided (from a prior `prepare_encode`), this
    function will reuse it instead of re-packing the meta dict — useful
    for fan-out to multiple publishers.
    """
    if meta_bytes is None:
        meta_bytes = _pack_meta(d.meta)
    body_size = _body_size(d)
    _HEADER.pack_into(view, 0, _MAGIC, _VERSION, d.dtype.value, len(meta_bytes), body_size)
    off = _HEADER_SIZE
    view[off : off + len(meta_bytes)] = meta_bytes
    off += len(meta_bytes)
    off = _write_body(d, view, off)
    return off


def _write_body(d: Data, view: memoryview, off: int) -> int:
    if d.dtype == DataType.ARRAY:
        a = np.ascontiguousarray(d.data)
        dtype_str = a.dtype.str.encode("ascii")
        _ARRAY_SUBHEADER.pack_into(view, off, a.ndim, len(dtype_str))
        off += _ARRAY_SUBHEADER.size
        view[off : off + len(dtype_str)] = dtype_str
        off += len(dtype_str)
        if a.ndim:
            struct.pack_into(f"<{a.ndim}I", view, off, *a.shape)
            off += 4 * a.ndim
        # zero-copy of the array body via numpy's buffer protocol. The
        # destination memoryview is a writable view onto the SHM slice (or
        # a bytearray); np.frombuffer creates an ndarray sharing that
        # memory, and slice-assignment performs a single memcpy from the
        # source heap buffer.
        if a.nbytes:
            dest = np.frombuffer(view, dtype=a.dtype, count=a.size, offset=off).reshape(a.shape)
            dest[...] = a
            off += a.nbytes
        return off
    if d.dtype == DataType.STRING:
        b = d.data.encode("utf-8")
        view[off : off + len(b)] = b
        return off + len(b)
    if d.dtype == DataType.TABLE:
        items = list(d.data.items())
        struct.pack_into("<I", view, off, len(items))
        off += 4
        for key, value in items:
            kb = key.encode("utf-8")
            struct.pack_into("<H", view, off, len(kb))
            off += 2
            view[off : off + len(kb)] = kb
            off += len(kb)
            value_size = data_byte_size(value)
            struct.pack_into("<I", view, off, value_size)
            off += 4
            encode_data_into(value, view[off : off + value_size])
            off += value_size
        return off
    raise ValueError(f"Unsupported dtype: {d.dtype}")


def encode_data(d: Data) -> bytes:
    """Allocate a fresh `bytes` buffer and encode `d` into it.

    Convenience wrapper for callers that don't have a pre-allocated
    buffer (the thread transport and unit tests). The IPC fast path
    uses `data_byte_size` + `encode_data_into` to write directly into
    an iceoryx2 SHM slice.
    """
    size = data_byte_size(d)
    buf = bytearray(size)
    written = encode_data_into(d, memoryview(buf))
    assert written == size, f"encode size mismatch: {written} != {size}"
    return bytes(buf)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode_data(buf: Union[bytes, bytearray, memoryview]) -> Data:
    """Decode an encoded byte buffer into a fresh `Data` object.

    Accepts any bytes-like (`bytes`, `bytearray`, or `memoryview`). For
    ARRAY data the returned `ndarray` owns its memory (a copy of the
    source bytes), so the caller may safely drop the underlying buffer
    afterwards — important when decoding from an iceoryx2 Sample whose
    SHM region is only valid while the Sample is held.
    """
    view = memoryview(buf)
    if len(view) < _HEADER_SIZE:
        raise ValueError(f"Buffer too small for header: {len(view)} bytes")
    magic, version, dtype_tag, meta_len, body_len = _HEADER.unpack_from(view, 0)
    if magic != _MAGIC:
        raise ValueError(f"Invalid Data frame: bad magic {bytes(magic)!r}")
    if version != _VERSION:
        raise ValueError(f"Unsupported frame version {version}")
    dtype = DataType(dtype_tag)
    off = _HEADER_SIZE
    meta = msgpack.unpackb(bytes(view[off : off + meta_len]), raw=False) if meta_len else {}
    if meta is None:
        meta = {}
    off += meta_len
    body = _read_body(dtype, view, off, body_len)
    return Data(dtype=dtype, data=body, meta=meta)


def _read_body(dtype: DataType, view: memoryview, off: int, body_len: int):
    end = off + body_len
    if dtype == DataType.ARRAY:
        ndim, dtype_str_len = _ARRAY_SUBHEADER.unpack_from(view, off)
        off += _ARRAY_SUBHEADER.size
        dtype_str = bytes(view[off : off + dtype_str_len]).decode("ascii")
        off += dtype_str_len
        if ndim:
            shape: Tuple[int, ...] = struct.unpack_from(f"<{ndim}I", view, off)
            off += 4 * ndim
        else:
            shape = ()
        np_dtype = np.dtype(dtype_str)
        nbytes = int(np.prod(shape)) * np_dtype.itemsize if shape else np_dtype.itemsize
        # Copy out of the source view so the resulting ndarray owns its
        # memory (the source may be an iceoryx2 Sample about to be released).
        arr = np.frombuffer(bytes(view[off : off + nbytes]), dtype=np_dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr
    if dtype == DataType.STRING:
        return bytes(view[off:end]).decode("utf-8")
    if dtype == DataType.TABLE:
        (n_entries,) = struct.unpack_from("<I", view, off)
        off += 4
        out = {}
        for _ in range(n_entries):
            (key_len,) = struct.unpack_from("<H", view, off)
            off += 2
            key = bytes(view[off : off + key_len]).decode("utf-8")
            off += key_len
            (value_len,) = struct.unpack_from("<I", view, off)
            off += 4
            out[key] = decode_data(view[off : off + value_len])
            off += value_len
        return out
    raise ValueError(f"Unsupported dtype: {dtype}")


# ---------------------------------------------------------------------------
# Message envelope — still vanilla msgpack (small, no zero-copy win)
# ---------------------------------------------------------------------------


def encode_message(m) -> bytes:
    return msgpack.packb({"type": m.type.value, "content": m.content}, default=_msgpack_default, use_bin_type=True)


def decode_message(buf: Union[bytes, bytearray, memoryview]):
    # Lazy import: keeps codec importable in contexts that mock Message.
    from goofi.message import Message, MessageType

    obj = msgpack.unpackb(bytes(buf), raw=False)
    return Message(type=MessageType(obj["type"]), content=obj["content"])
