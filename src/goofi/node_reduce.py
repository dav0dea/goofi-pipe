"""Node-side thalamus reduction: fold a Data to a per-axis ViewSpec (numpy only).

Pure functions, no transport/manager/frontend coupling. Imported by the node's
reducer thread (Phase 2). FAIL-OPEN by contract: any guard trip or exception in
reduce_for_view returns the input Data unreduced.

A ViewSpec lists axes to reduce, each with a `max` (target entries) and a
`method` ('envelope' | 'subsample' | 'area'). reduce_for_view composes them and
co-reduces each reduced axis's coord (meta['channels']['dimD']) with the SAME
transform, so the reduced Data satisfies Data.__post_init__'s per-axis
coord-length assertion (data.py). Each reduced axis records reconstruction info in
meta['reduced'][str(axis)] so the metadata inspector can show the true original meta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from goofi.data import Data, DataType

_METHODS = ("envelope", "subsample", "area")
_RICHNESS = {"envelope": 3, "area": 2, "subsample": 1}  # richest wins on fold conflict
_ORIG_COORD_CAP = 4096  # carry orig_coord verbatim only for subsample axes <= this

# Per-viewer-kind default axes (used by the manager relay to seed a ViewSpec from
# the URL `kind` before the browser sends a capacity-derived one). Mirrors the
# per-kind axis table in the spec (§6.3). Non-reducible kinds reduce nothing.
_DEFAULT_VIEW_AXES = {
    "line": [{"axis": -1, "max": 2000, "method": "envelope"}],
    "image": [{"axis": 0, "max": 720, "method": "area"}, {"axis": 1, "max": 1280, "method": "area"}],
    "trajectory": [{"axis": 0, "max": 5000, "method": "subsample"}],
}


@dataclass(frozen=True)
class AxisSpec:
    axis: int  # may be negative; canonicalized in reduce_for_view
    max: int  # target entries on this axis (>=1)
    method: str  # 'envelope' | 'subsample' | 'area'


@dataclass(frozen=True)
class ViewSpec:
    axes: Tuple[AxisSpec, ...] = ()  # axes to reduce; unlisted axes untouched
    version: int = 0  # client ordering only; node ignores


def viewspec_from_dict(d: dict) -> ViewSpec:
    axes = []
    for a in (d.get("axes") or []):
        if not isinstance(a, dict):
            continue
        try:
            ax = int(a.get("axis"))
            mx = max(1, int(a.get("max")))
        except Exception:
            continue
        method = a.get("method")
        if method not in _METHODS:
            continue
        axes.append(AxisSpec(axis=ax, max=mx, method=method))
    try:
        ver = int(d.get("version", 0) or 0)
    except Exception:
        ver = 0
    return ViewSpec(axes=tuple(axes), version=ver)


# ---------------------------------------------------------------------------
# Per-axis reduction primitives (numpy only)
# ---------------------------------------------------------------------------


def _subsample_idx(n: int, m: int) -> np.ndarray:
    """Linspace indices into [0, n); len <= min(n, m). The indices are already
    ascending, so np.unique both de-dups (round() can collide for tiny n) and keeps
    order in one pass — no separate sort needed."""
    m = min(max(1, m), n)
    idx = np.linspace(0, n - 1, m).round().astype(int)
    return np.unique(idx)


def _envelope(x: np.ndarray, axis: int, w: int):
    """Min/max envelope along `axis`: returns (env[..,2*w,..], bin_centers[w] int idx).
    Caller applies the 2x ratio skip guard before invoking (see _apply_axis)."""
    n = x.shape[axis]
    w = min(max(1, w), n)
    edges = np.linspace(0, n, w + 1).astype(int)
    xs = np.moveaxis(x, axis, -1)  # sample axis last
    out = np.empty(xs.shape[:-1] + (2 * w,), dtype=xs.dtype)
    centers = np.empty(w, dtype=int)
    for b in range(w):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        seg = xs[..., lo:hi]
        out[..., 2 * b] = seg.min(axis=-1)
        out[..., 2 * b + 1] = seg.max(axis=-1)
        centers[b] = (lo + hi - 1) // 2
    return np.ascontiguousarray(np.moveaxis(out, -1, axis)), centers


def _area_axis(x: np.ndarray, axis: int, m: int):
    """Block-mean (area) downscale along ONE axis to min(m, n) bins (numpy only).
    Per-axis block-mean is separable and numerically identical to a true 2-D block
    mean for non-divisor ratios. Returns (out, bin_centers[m] int idx)."""
    n = x.shape[axis]
    m = min(max(1, m), n)
    edges = np.linspace(0, n, m + 1).astype(int)
    counts = np.maximum(1, np.diff(edges))
    f = x.astype(np.float32)
    summed = np.add.reduceat(f, edges[:-1], axis=axis)
    shape = [1] * x.ndim
    shape[axis] = m
    out = summed / counts.reshape(shape)
    centers = ((edges[:-1] + np.maximum(edges[:-1] + 1, edges[1:]) - 1) // 2).astype(int)
    return np.ascontiguousarray(out.astype(x.dtype)), centers


# ---------------------------------------------------------------------------
# Single-axis application + coord co-reduction
# ---------------------------------------------------------------------------


def _set_coord(meta: dict, dim: int, new_coord: Optional[list]) -> None:
    ch = meta.setdefault("channels", {})
    k = f"dim{dim}"
    if new_coord is None:
        ch.pop(k, None)  # axis lost its coord -> drop (body still valid)
    else:
        ch[k] = list(new_coord)


def _apply_axis(arr: np.ndarray, axis: int, a: AxisSpec, meta: dict):
    """Apply one axis reduction; co-reduce that axis's coord; return (out, info).
    info = {orig_len, method, orig_coord?}; returns (arr, None) for a no-op skip."""
    orig_len = int(arr.shape[axis])
    ch = meta.get("channels") or {}
    coord = ch.get(f"dim{axis}")
    coord = list(coord) if isinstance(coord, (list, tuple)) else None

    if a.method == "envelope":
        w = min(max(1, a.max), orig_len)
        if orig_len < 2 * w:  # would not shrink >=2x -> skip this axis entirely
            return arr, None
        out, centers = _envelope(arr, axis, a.max)
        new_coord = (list(np.repeat(np.asarray(coord)[centers], 2))
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        return out, {"orig_len": orig_len, "method": "envelope"}

    if a.method == "subsample":
        if min(max(1, a.max), orig_len) >= orig_len:  # already fits -> skip (no-op take)
            return arr, None
        idx = _subsample_idx(orig_len, a.max)
        out = np.ascontiguousarray(np.take(arr, idx, axis=axis))
        new_coord = ([coord[i] for i in idx]
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        info = {"orig_len": orig_len, "method": "subsample"}
        if coord is not None and len(coord) == orig_len and orig_len <= _ORIG_COORD_CAP:
            info["orig_coord"] = list(coord)
        return out, info

    if a.method == "area":
        if min(max(1, a.max), orig_len) >= orig_len:  # already fits -> skip the no-op copy
            return arr, None
        out, centers = _area_axis(arr, axis, a.max)
        new_coord = ([coord[i] for i in centers]
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        return out, {"orig_len": orig_len, "method": "area"}

    return arr, None  # unknown method -> no-op (defensive; viewspec_from_dict filters)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def default_viewspec_for_kind(kind: str) -> dict:
    """Seed ViewSpec dict for a viewer `kind` (manager relay default). Unknown /
    non-reducible kinds (string/table/topomap/...) reduce nothing."""
    return {"axes": [dict(a) for a in _DEFAULT_VIEW_AXES.get(kind, [])], "version": 0}


def fold_viewspecs(specs: list) -> dict:
    """Fold N per-axis ViewSpec dicts into one (manager folds all browsers of a
    slot). Richest-wins per axis: max() of the per-axis `max`, richest method
    (envelope > area > subsample). Keyed by the raw `axis` value — clients of one
    slot share an axis convention. `version` = max across inputs (client ordering)."""
    by_axis: Dict[int, dict] = {}
    ver = 0
    for s in specs or []:
        if not isinstance(s, dict):
            continue
        try:
            ver = max(ver, int(s.get("version", 0) or 0))
        except Exception:
            pass
        for a in (s.get("axes") or []):
            if not isinstance(a, dict):
                continue
            try:
                ax = int(a.get("axis"))
                mx = max(1, int(a.get("max")))
            except Exception:
                continue
            method = a.get("method")
            if method not in _METHODS:
                continue
            cur = by_axis.get(ax)
            if cur is None:
                by_axis[ax] = {"axis": ax, "max": mx, "method": method}
            else:
                cur["max"] = max(cur["max"], mx)
                if _RICHNESS.get(method, 0) > _RICHNESS.get(cur["method"], 0):
                    cur["method"] = method
    return {"axes": [by_axis[k] for k in sorted(by_axis.keys())], "version": ver}


def reduce_for_view(data: Data, spec: Optional[ViewSpec]) -> Data:
    """Return a (possibly) smaller Data for `spec` by composing PER-AXIS reductions.
    FAIL-OPEN: any guard/exception returns `data` unreduced. NEVER mutates `data`."""
    if spec is None or not spec.axes:
        return data
    try:
        if data.dtype != DataType.ARRAY:  # STRING/TABLE -> passthrough
            return data
        arr = data.data
        if not hasattr(arr, "ndim") or arr.ndim == 0:
            return data
        ndim = arr.ndim

        # Canonicalize axes to positive and de-dup. Two raw axes can collapse onto
        # one canonical axis (a 1-D line sends channel axis 0 AND sample axis -1,
        # both -> 0). Fold collisions exactly like fold_viewspecs: richest method
        # wins (so a waveform keeps its peak-preserving envelope over an aliased
        # subsample) AND the larger cap wins (max-of-maxes). Positional last-wins
        # used to drop peaks; richness-only used to drop the larger cap.
        by_axis = {}
        for a in spec.axes:
            c = a.axis % ndim
            prev = by_axis.get(c)
            if prev is None:
                by_axis[c] = a
            else:
                method = a.method if _RICHNESS.get(a.method, 0) >= _RICHNESS.get(prev.method, 0) else prev.method
                by_axis[c] = AxisSpec(axis=prev.axis, max=max(prev.max, a.max), method=method)
        if not by_axis:
            return data

        # Aspect-preserving image downscale: when 2+ axes use 'area' (an image's
        # H and W), capping each independently to the viewer's pixel box would
        # squash a non-square source into the box's aspect. Apply ONE uniform
        # downscale factor (min over the area axes, never upscale) so the reduced
        # image keeps the source aspect ratio; the viewer letterboxes it. (Aspect
        # is exact for axes large enough that round(orig*scale) actually shrinks;
        # a degenerate ~2-3px axis can round back to orig and drift sub-pixel.)
        area_axes = [cax for cax, a in by_axis.items() if a.method == "area"]
        if len(area_axes) >= 2:
            scale = min(1.0, min(by_axis[cax].max / arr.shape[cax] for cax in area_axes))
            for cax in area_axes:
                target = max(1, round(arr.shape[cax] * scale))
                by_axis[cax] = AxisSpec(axis=by_axis[cax].axis, max=target, method="area")

        # shallow-copy meta; deep-copy the channels sub-dict before edits (never mutate input)
        new_meta = dict(data.meta)
        ch_src = new_meta.get("channels") or {}
        new_meta["channels"] = {k: list(v) if isinstance(v, (list, tuple)) else v
                                for k, v in ch_src.items()}
        reduced_info = {}

        out = arr
        # All three methods preserve ndim, so descending order keeps indices valid.
        for cax in sorted(by_axis.keys(), reverse=True):
            new_out, info = _apply_axis(out, cax, by_axis[cax], new_meta)
            if info is None:  # axis skipped (e.g. envelope <2x) -> leave out as-is
                continue
            out = new_out
            reduced_info[str(cax)] = info

        if not reduced_info:  # nothing actually reduced -> passthrough the original
            return data

        new_meta["reduced"] = reduced_info
        out = np.ascontiguousarray(out)
        return Data(data.dtype, out, new_meta)  # constructor is the final net
    except Exception:
        return data  # FAIL-OPEN
