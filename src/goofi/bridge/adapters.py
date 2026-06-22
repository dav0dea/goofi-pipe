"""Viewer adapters — per-kind float→view representation at the bridge boundary.

A node emits pure-float ``Data``; the data plane decodes it once per slot and, for
each distinct viewer kind subscribed, calls ``adapt(data, kind)`` to produce the
representation that kind renders best on a lower-bit transport:

    image                       -> uint8   (RGB/RGBA clamp; grayscale normalize)
    line / trajectory / topomap -> float16
    string / table / raw        -> passthrough (unchanged)

Float range/stats travel under the namespaced ``meta["__view__"]`` key so the
viewer's value range and the metadata inspector stay float-accurate; uint8/float16
never re-enter node processing, persistence, or the inspector. Adapting never
mutates the input ``Data`` — node meta is deep-copied before ``__view__`` is added.

See ``docs/superpowers/specs/2026-06-21-viewer-adapters-design.md`` (backlog #3).
"""
from __future__ import annotations

import copy
from typing import Callable, Dict

import numpy as np

from goofi.data import Data, DataType
from goofi.image_utils import as_uint8

_FLAT_EPS = 1e-12


def view_stats(arr: np.ndarray) -> Dict[str, float]:
    """``{min, mean, max}`` computed on the (float) array."""
    a = np.asarray(arr, dtype=np.float64)
    return {"min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}


def is_renderable(kind: str, shape) -> bool:
    """Whether ``kind`` can draw an array of ``shape``.

    Mirror of the frontend ``isRenderable`` (``frontend/src/lib/viewers/kind.ts``)
    so the bridge and the viewer agree on when to fall back to a text summary.
    """
    n = len(shape)
    if kind == "line":
        return n <= 3
    if kind == "image":
        return n == 2 or (n == 3 and shape[2] in (1, 2, 3, 4))
    if kind == "trajectory":
        return n == 2 and shape[0] >= 2
    if kind == "topomap":
        return n == 1
    return True


def _summary(data: Data) -> Data:
    """Non-renderable fallback: a bodyless frame carrying a float summary."""
    arr = np.asarray(data.data)
    summary = {"shape": list(arr.shape), "dtype": arr.dtype.str, **view_stats(arr)}
    return Data(DataType.ARRAY, np.zeros(0, dtype=np.float32), {"__view__": {"summary": summary}})


def adapt_image(data: Data) -> Data:
    arr = np.asarray(data.data)
    if not is_renderable("image", arr.shape):
        return _summary(data)
    fmin, fmax = float(arr.min()), float(arr.max())
    stats = view_stats(arr)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        # RGB/RGBA: straight [0,1]->[0,255] clamp, no normalization (preserves colour).
        body = as_uint8(arr)
    else:
        # grayscale: normalize [fmin,fmax]->[0,255] so the colormap uses the full range.
        span = fmax - fmin
        if span < _FLAT_EPS:
            span = 1.0
        norm = (arr.astype(np.float32) - fmin) / span
        body = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    meta = copy.deepcopy(data.meta)
    meta["__view__"] = {"range": [fmin, fmax], "stats": stats}
    return Data(DataType.ARRAY, body, meta)


def _make_float16_adapter(kind: str) -> Callable[[Data], Data]:
    def _adapt(data: Data) -> Data:
        arr = np.asarray(data.data)
        if not is_renderable(kind, arr.shape):
            return _summary(data)
        stats = view_stats(arr)  # on the float32 array, before the lossy downcast
        meta = copy.deepcopy(data.meta)
        meta["__view__"] = {"stats": stats}
        return Data(DataType.ARRAY, arr.astype(np.float16), meta)

    return _adapt


def adapt_raw(data: Data) -> Data:
    """Passthrough for string/table/unknown kinds — the node Data is forwarded as-is."""
    return data


ADAPTERS: Dict[str, Callable[[Data], Data]] = {
    "image": adapt_image,
    "line": _make_float16_adapter("line"),
    "trajectory": _make_float16_adapter("trajectory"),
    "topomap": _make_float16_adapter("topomap"),
    "string": adapt_raw,
    "table": adapt_raw,
    "raw": adapt_raw,
}


def adapt(data: Data, kind: str) -> Data:
    """Convert ``data`` to the representation ``kind`` needs (``raw`` fallback)."""
    return ADAPTERS.get(kind, adapt_raw)(data)
