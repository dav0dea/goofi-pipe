# Node-Side Thalamus Reduction + Manager Relay (Option C) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce each viewer frame to display size *inside the producing node* (a node-side "thalamus") and have the manager forward the already-reduced, already-encoded frames to browsers verbatim — removing the manager's full-Data decode→re-encode from the viewer path without moving to per-node P2P sockets.

**Architecture:** Three work areas. (1) A pure-numpy reduction engine (`node_reduce.py`) that folds a `Data` to a per-axis `ViewSpec`. (2) A node-side viewer-publish path: a `viewer_count` gate, a dedicated reducer thread fed by an O(1) `offer()` from the processing loop, and reduced frames published on a *separate* iceoryx2 service. (3) The manager/bridge relay: it folds browser ViewSpecs per `(node, slot)`, pushes the folded spec to the node over the ctrl plane, subscribes to the node's *reduced* service, and fans frames to browsers raw (no decode/re-encode). The node↔node full-Data SHM path and `codec.py`/`transport.py` are untouched.

**Tech Stack:** Python 3.12, numpy, iceoryx2 (existing transport), aiohttp (existing bridge), pytest. Frontend: SvelteKit, Svelte 5 runes, TypeScript, vitest, Playwright.

**Source spec:** `docs/p2p-data-thalamus-spec.md` (written for full P2P; this plan adapts §5–§6 and replaces §4/§7-host-scope/§9-delete with the **Option C relay** chosen on 2026-06-25). Where this plan and the spec disagree, **this plan wins** (it reflects the Option C decision).

## Global Constraints

- `node_reduce.py`: **numpy only**. No PIL, scipy, cv2, torch. All reductions pure-numpy.
- Python style: 4-space indent, double-quoted strings, match surrounding code. **Never run Prettier/black** on this repo; hand-match.
- **TDD is the Iron Law** (CLAUDE.md): write the failing test, watch it fail for the right reason, write the minimal code to pass, refactor green. No production code without a failing test first.
- `reduce_for_view` invariants (verbatim from spec §6.2): **FAIL-OPEN** (any guard/exception returns the input `Data` unreduced); **NEVER mutates** the input `Data` or its arrays/meta; every produced array is a **fresh `np.ascontiguousarray`**; passthrough returns the input object itself.
- The reduced `Data` must satisfy `Data.__post_init__` (`data.py:110-116`): for every axis `d` with a coord list `meta["channels"]["dim{d}"]`, `len(coord) == reduced.shape[d]`. Co-reduce every reduced axis's coord with the same transform.
- `.venv/bin/python -m pytest tests/` must stay green after every backend task (currently ~990 passing).
- Commit in small focused steps. Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Work on a feature branch `feat/node-thalamus-reduction` off `frontend` (do **not** touch `main`).

---

## File Structure

| file | responsibility | phase |
|---|---|---|
| `src/goofi/node_reduce.py` (NEW) | pure-numpy reduction engine: `AxisSpec`, `ViewSpec`, `viewspec_from_dict`, `reduce_for_view`, `_apply_axis`, `_envelope`, `_area_axis`, `_subsample_idx`, `_set_coord` | 1 |
| `tests/test_node_reduce.py` (NEW) | unit tests for the reduction engine | 1 |
| `src/goofi/node_helpers.py` (MOD) | `OutputSlot.viewer_count` int field + lazy `viewer_lock` property | 2 |
| `src/goofi/node.py` (MOD) | OR-gate (`subscriber_count==0 and viewer_count==0`); split-encode (full→SHM gated on `subscriber_count>0`; viewer path = one `offer()`); reducer thread bootstrap; `SET_VIEWSPEC` ctrl handling; reduced-frame publish on viewer service | 2 |
| `src/goofi/node_viewer.py` (NEW) | the reducer subsystem: `offer`, `_snapshot_for_offer`, `_reducer_loop`, per-`(node,slot)` folded spec store, reduced-frame publish via the existing transport; `_reset_for_tests` | 2 |
| `src/goofi/bridge/data.py` (MOD) | relay: subscribe to node's *reduced* service, forward raw; fold browser ViewSpecs per `(node,slot)`; push folded spec to node over ctrl; viewer register/unregister gating | 3 |
| `src/goofi/node_helpers.py` (MOD) | `NodeRef.set_viewspec(slot, spec)` ctrl send; reduced-service subscription helper | 3 |
| frontend `viewers/*`, `stores/thalamus.svelte.ts` (NEW), `api/data.ts` (MOD), `editor/MetadataPanel.svelte` (MOD) | per-axis ViewSpec from viewer capacity; send up; reduction-aware metadata; envelope band | 4 |

**This document fully details Phase 1.** Phases 2–4 are specified as a roadmap at the end and will each get their own detailed plan (with exact line numbers and code) once Phase 1 lands and its public API is fixed — this respects the writing-plans scope rule (one subsystem per detailed plan) and avoids pinning line numbers in code we haven't yet shaped.

---

## Phase 1 — Reduction engine (`node_reduce.py`)

### Task 1: `AxisSpec`, `ViewSpec`, `viewspec_from_dict`

**Files:**
- Create: `src/goofi/node_reduce.py`
- Test: `tests/test_node_reduce.py`

**Interfaces:**
- Produces: `AxisSpec(axis:int, max:int, method:str)` frozen dataclass; `ViewSpec(axes:Tuple[AxisSpec,...]=(), version:int=0)` frozen dataclass; `viewspec_from_dict(d:dict) -> ViewSpec` (tolerant parser: skips non-dict axes, clamps `max>=1`, drops methods not in `("envelope","subsample","area")`, defaults `version` to 0 on any error).
- Module constants: `_METHODS=("envelope","subsample","area")`, `_ORIG_COORD_CAP=4096`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_node_reduce.py
import numpy as np
import pytest
from goofi.data import Data, DataType
from goofi.node_reduce import AxisSpec, ViewSpec, viewspec_from_dict


def test_viewspec_from_dict_parses_valid_axes():
    spec = viewspec_from_dict(
        {"axes": [{"axis": -1, "max": 1600, "method": "envelope"}], "version": 3}
    )
    assert spec.version == 3
    assert spec.axes == (AxisSpec(axis=-1, max=1600, method="envelope"),)


def test_viewspec_from_dict_is_tolerant():
    spec = viewspec_from_dict(
        {
            "axes": [
                {"axis": 0, "max": 0, "method": "subsample"},      # max clamped to >=1
                {"axis": 1, "max": 10, "method": "bogus"},         # unknown method dropped
                "not-a-dict",                                       # skipped
                {"axis": 2, "max": 5, "method": "area"},
            ],
            "version": "nope",                                     # bad version -> 0
        }
    )
    assert spec.version == 0
    assert spec.axes == (
        AxisSpec(axis=0, max=1, method="subsample"),
        AxisSpec(axis=2, max=5, method="area"),
    )


def test_viewspec_empty_default():
    assert viewspec_from_dict({}) == ViewSpec()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'goofi.node_reduce'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/goofi/node_reduce.py
"""Node-side thalamus reduction: fold a Data to a per-axis ViewSpec (numpy only).

Pure functions, no transport/manager/frontend coupling. Imported by the node's
reducer thread (Phase 2). FAIL-OPEN by contract: any guard trip or exception in
reduce_for_view returns the input Data unreduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from goofi.data import Data, DataType

_METHODS = ("envelope", "subsample", "area")
_ORIG_COORD_CAP = 4096  # carry orig_coord verbatim only for subsample axes <= this


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/goofi/node_reduce.py tests/test_node_reduce.py
git commit -m "feat(reduce): ViewSpec/AxisSpec + tolerant viewspec_from_dict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_subsample_idx`

**Interfaces:**
- Produces: `_subsample_idx(n:int, m:int) -> np.ndarray` — unique-preserving `linspace(0,n-1)` integer indices, ascending, length `min(n,m)`, no duplicates.

- [ ] **Step 1: Write the failing test**

```python
from goofi.node_reduce import _subsample_idx


def test_subsample_idx_picks_spread_indices():
    idx = _subsample_idx(100, 5)
    assert list(idx) == [0, 25, 50, 74, 99]


def test_subsample_idx_caps_at_n_and_is_unique():
    idx = _subsample_idx(3, 10)         # m > n -> at most n, no dups
    assert list(idx) == [0, 1, 2]
    assert len(set(idx)) == len(idx)
    assert list(idx) == sorted(idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py::test_subsample_idx_picks_spread_indices -v`
Expected: FAIL — `ImportError: cannot import name '_subsample_idx'`

- [ ] **Step 3: Write minimal implementation** (append to `node_reduce.py`)

```python
def _subsample_idx(n: int, m: int) -> np.ndarray:
    """Unique-preserving linspace indices; len = min(n, m)."""
    m = min(max(1, m), n)
    idx = np.linspace(0, n - 1, m).round().astype(int)
    # unique-preserving (linspace.round can collide for tiny n); keep order, drop dups
    _, keep = np.unique(idx, return_index=True)
    return idx[np.sort(keep)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/goofi/node_reduce.py tests/test_node_reduce.py
git commit -m "feat(reduce): _subsample_idx unique-preserving linspace indices

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `_envelope`

**Interfaces:**
- Produces: `_envelope(x:np.ndarray, axis:int, w:int) -> (np.ndarray, np.ndarray)` — min/max envelope along `axis`: returns `(env, centers)` where `env` has length `2*min(w,n)` on `axis` (per-bin min,max interleaved) and `centers` is the `w` integer bin-center indices on the original axis. Output is C-contiguous.

- [ ] **Step 1: Write the failing test**

```python
from goofi.node_reduce import _envelope


def test_envelope_interleaves_min_max_per_bin():
    x = np.array([0.0, 5.0, 1.0, 4.0, 2.0, 3.0], dtype=np.float32)  # n=6
    env, centers = _envelope(x, axis=0, w=3)                        # 3 bins of 2
    assert env.shape == (6,)                                        # 2*w
    assert list(env) == [0.0, 5.0, 1.0, 4.0, 2.0, 3.0]             # (min,max) per pair
    assert list(centers) == [0, 2, 4]


def test_envelope_preserves_other_axes_2d():
    x = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)            # (C=2, N=8)
    env, centers = _envelope(x, axis=1, w=4)
    assert env.shape == (2, 8)                                      # rows kept, 2*4 on axis1
    assert env.flags["C_CONTIGUOUS"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py::test_envelope_interleaves_min_max_per_bin -v`
Expected: FAIL — `ImportError: cannot import name '_envelope'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/goofi/node_reduce.py tests/test_node_reduce.py
git commit -m "feat(reduce): _envelope min/max envelope along an axis

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `_area_axis`

**Interfaces:**
- Produces: `_area_axis(x:np.ndarray, axis:int, m:int) -> (np.ndarray, np.ndarray)` — block-mean downscale along ONE axis to `min(m,n)` bins via `np.add.reduceat` (handles non-divisor ratios); returns `(out, centers)` with `out` C-contiguous in `x`'s dtype and `centers` the `m` integer bin-center indices.

- [ ] **Step 1: Write the failing test**

```python
from goofi.node_reduce import _area_axis


def test_area_axis_block_mean_divisor():
    x = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)   # n=4 -> 2 bins
    out, centers = _area_axis(x, axis=0, m=2)
    assert list(out) == [1.0, 5.0]                          # mean(0,2), mean(4,6)
    assert list(centers) == [0, 2]


def test_area_axis_equals_true_2d_block_mean_nondivisor():
    rng = np.random.default_rng(0)
    img = rng.random((7, 5)).astype(np.float32)            # non-divisor ratios
    a, _ = _area_axis(img, 0, 3)
    a, _ = _area_axis(a, 1, 2)                              # separable compose
    # reference: explicit 2-D block mean over the same linspace edges
    ei = np.linspace(0, 7, 4).astype(int)
    ej = np.linspace(0, 5, 3).astype(int)
    ref = np.empty((3, 2), dtype=np.float32)
    for i in range(3):
        for j in range(2):
            ref[i, j] = img[ei[i]:max(ei[i] + 1, ei[i + 1]),
                            ej[j]:max(ej[j] + 1, ej[j + 1])].mean()
    assert np.max(np.abs(a - ref)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py::test_area_axis_block_mean_divisor -v`
Expected: FAIL — `ImportError: cannot import name '_area_axis'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/goofi/node_reduce.py tests/test_node_reduce.py
git commit -m "feat(reduce): _area_axis separable block-mean downscale

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `_set_coord` + `_apply_axis` (single-axis reduction with coord co-reduction)

**Interfaces:**
- Consumes: `_envelope`, `_area_axis`, `_subsample_idx` (Tasks 2–4).
- Produces:
  - `_set_coord(meta:dict, dim:int, new_coord:Optional[list]) -> None` — sets `meta["channels"]["dim{dim}"]` to `new_coord`, or pops it when `new_coord is None`.
  - `_apply_axis(arr:np.ndarray, axis:int, a:AxisSpec, meta:dict) -> (np.ndarray, Optional[dict])` — applies one axis reduction, co-reduces that axis's coord in `meta`, returns `(out, info)` where `info = {"orig_len":N, "method":..., "orig_coord"?:[...]}`. Returns `(arr, None)` (no-op, no meta change) when an **envelope** axis would not shrink by ≥2× (`orig_len < 2*min(a.max, orig_len)`).

- [ ] **Step 1: Write the failing test**

```python
from goofi.node_reduce import _apply_axis


def test_apply_axis_envelope_coreduces_coord_and_records_info():
    arr = np.arange(8, dtype=np.float32)                 # n=8
    meta = {"channels": {"dim0": list(range(8))}}
    out, info = _apply_axis(arr, 0, AxisSpec(0, 2, "envelope"), meta)
    assert out.shape == (4,)                             # 2*w
    assert len(meta["channels"]["dim0"]) == 4            # co-reduced -> matches body
    assert info == {"orig_len": 8, "method": "envelope"}


def test_apply_axis_subsample_carries_small_orig_coord():
    arr = np.arange(6, dtype=np.float32)
    meta = {"channels": {"dim0": ["a", "b", "c", "d", "e", "f"]}}
    out, info = _apply_axis(arr, 0, AxisSpec(0, 3, "subsample"), meta)
    assert out.shape == (3,)
    assert len(meta["channels"]["dim0"]) == 3
    assert info["method"] == "subsample"
    assert info["orig_coord"] == ["a", "b", "c", "d", "e", "f"]  # <= cap


def test_apply_axis_envelope_skips_when_not_2x_smaller():
    arr = np.arange(8, dtype=np.float32)
    meta = {"channels": {"dim0": list(range(8))}}
    out, info = _apply_axis(arr, 0, AxisSpec(0, 6, "envelope"), meta)  # 2*6=12 > 8
    assert out is arr                                    # untouched
    assert info is None
    assert meta["channels"]["dim0"] == list(range(8))    # coord untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py::test_apply_axis_envelope_coreduces_coord_and_records_info -v`
Expected: FAIL — `ImportError: cannot import name '_apply_axis'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
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
        out, centers = _area_axis(arr, axis, a.max)
        new_coord = ([coord[i] for i in centers]
                     if coord is not None and len(coord) == orig_len else None)
        _set_coord(meta, axis, new_coord)
        return out, {"orig_len": orig_len, "method": "area"}

    return arr, None  # unknown method -> no-op (defensive; viewspec_from_dict filters)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/goofi/node_reduce.py tests/test_node_reduce.py
git commit -m "feat(reduce): _apply_axis single-axis reduction + coord co-reduction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `reduce_for_view` (compose per-axis; fail-open; meta; Data construction)

**Interfaces:**
- Consumes: `_apply_axis` (Task 5), `ViewSpec` (Task 1).
- Produces: `reduce_for_view(data:Data, spec:Optional[ViewSpec]) -> Data` — composes per-axis reductions (descending canonical-axis order), records `meta["reduced"][str(axis)]=info` per reduced axis, constructs a fresh reduced `Data`. Obeys all Global-Constraint invariants (fail-open, never-mutate, fresh contiguous, passthrough returns input).

- [ ] **Step 1: Write the failing test**

```python
from goofi.node_reduce import reduce_for_view


def test_reduce_for_view_envelope_1d_full():
    n = 10000
    data = Data(DataType.ARRAY, np.linspace(-1, 1, n).astype(np.float32),
                {"channels": {"dim0": list(range(n))}})
    spec = viewspec_from_dict({"axes": [{"axis": -1, "max": 1000, "method": "envelope"}]})
    red = reduce_for_view(data, spec)
    assert red.data.shape == (2000,)                       # 2*W
    assert red.meta["reduced"]["0"] == {"orig_len": n, "method": "envelope"}
    assert len(red.meta["channels"]["dim0"]) == 2000        # constructor would assert otherwise


def test_reduce_for_view_does_not_mutate_input():
    n = 4096
    arr = np.linspace(0, 1, n).astype(np.float32)
    data = Data(DataType.ARRAY, arr, {"channels": {"dim0": list(range(n))}})
    before = arr.copy()
    reduce_for_view(data, viewspec_from_dict({"axes": [{"axis": 0, "max": 256, "method": "envelope"}]}))
    assert np.array_equal(arr, before)                      # input array untouched
    assert len(data.meta["channels"]["dim0"]) == n          # input meta untouched


def test_reduce_for_view_2d_line_both_axes():
    C, N = 64, 5000
    data = Data(DataType.ARRAY, np.zeros((C, N), dtype=np.float32),
                {"channels": {"dim0": [f"ch{i}" for i in range(C)], "dim1": list(range(N))}})
    spec = viewspec_from_dict({"axes": [
        {"axis": 0, "max": 8, "method": "subsample"},
        {"axis": -1, "max": 800, "method": "envelope"},
    ]})
    red = reduce_for_view(data, spec)
    assert red.data.shape == (8, 1600)                      # channel-cap + 2*W envelope
    assert len(red.meta["channels"]["dim0"]) == 8
    assert len(red.meta["channels"]["dim1"]) == 1600
    assert set(red.meta["reduced"].keys()) == {"0", "1"}    # both axes recorded, canonical


def test_reduce_for_view_fails_open_on_bad_input():
    data = Data(DataType.ARRAY, np.zeros((4,), dtype=np.float32), {})
    # axis 5 % 1 == 0 is fine; instead force an internal error via a non-array dtype path:
    s = Data(DataType.STRING, "hello", {})
    assert reduce_for_view(s, viewspec_from_dict({"axes": [{"axis": 0, "max": 2, "method": "area"}]})) is s


def test_reduce_for_view_none_or_empty_spec_passthrough():
    data = Data(DataType.ARRAY, np.zeros((4,), dtype=np.float32), {})
    assert reduce_for_view(data, None) is data
    assert reduce_for_view(data, ViewSpec()) is data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py::test_reduce_for_view_envelope_1d_full -v`
Expected: FAIL — `ImportError: cannot import name 'reduce_for_view'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
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

        # Canonicalize axes to positive; de-dup (last spec for an axis wins).
        by_axis = {}
        for a in spec.axes:
            by_axis[a.axis % ndim] = a
        if not by_axis:
            return data

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_node_reduce.py -v`
Expected: PASS (all node_reduce tests)

- [ ] **Step 5: Run the full backend suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS (~990, same as before; `node_reduce` is new + isolated)

- [ ] **Step 6: Commit**

```bash
git add src/goofi/node_reduce.py tests/test_node_reduce.py
git commit -m "feat(reduce): reduce_for_view composes per-axis reductions, fail-open

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phases 2–4 — Roadmap (detailed plans to follow)

> Each gets its own detailed TDD plan once the prior phase's public API is fixed. Summarized here so the whole Option-C arc is visible.

### Phase 2 — Node viewer-publish path
- **`OutputSlot.viewer_count`** (plain int, picklable) + lazy `viewer_lock` property (spec §5.1 — pickle-safe, no Lock field). *Test:* `pickle.dumps(OutputSlot(DataType.ARRAY))` + `copy.deepcopy` succeed.
- **OR-gate + split-encode** at the processing loop: gate on `subscriber_count==0 and viewer_count==0`; full encode→SHM guarded on `subscriber_count>0`; viewer path is one `offer()` line (spec §5.2-5.3). *Test:* a slot with `viewer_count=1, subscriber_count=0` ticks and calls `offer` (monkeypatched), does **not** run the full encode.
- **`node_viewer.py`**: `_snapshot_for_offer` (node-thread private copy, spec §6.6), `offer`, one `goofi-data-reducer` thread (`_reducer_loop`: pop latest-wins `_pending`, `reduce_for_view`, `encode_data`, publish on the slot's **reduced** iceoryx2 service), per-`(node,slot)` folded-spec store, `evict_slot`, `_reset_for_tests`. *Test:* offer→reduced frame published; `LatentRotator`-shaped in-place mutator can't crash the reducer; node↔node bytes unaffected when the reducer is forced to raise; latest-wins (two rapid offers → only latest reduced).
- **Reduced iceoryx2 service**: add a `data_service_name(node, slot) + ".view"` service (new `transport.py` helper name only — no transport mechanics change). The node publishes reduced frames here; node↔node full frames stay on the existing service.
- **`SET_VIEWSPEC` ctrl message**: node messaging loop stores the folded `ViewSpec` for a slot (last-received-wins). *Test:* sending `SET_VIEWSPEC` updates the slot's spec the reducer reads.

### Phase 3 — Manager/bridge relay (modify `bridge/data.py`, do **not** delete it)
- `_SlotMux` subscribes to the node's **reduced** `.view` service (via a new `NodeRef.open_reduced_subscriber`) and forwards frames **raw** (no `adapt`/`encode_data`). *Test:* a frame published on the reduced service reaches the browser WS byte-identical.
- **Viewer registration drives `viewer_count`** + idle-wake: on first browser for a `(node,slot)`, send a viewer-register ctrl msg (bumps `viewer_count`, wakes the node); on last close, unregister (decrement, `evict_slot`). Reuse the existing on-demand lifecycle in `DataHub.handler`.
- **ViewSpec fold**: `DataHub` collects each browser's per-axis ViewSpec (inbound `{op:"view"}` text on the `/data` WS), folds richest-wins per `(node,slot)` (debounce), and pushes the folded spec to the node via `NodeRef.set_viewspec` (`SET_VIEWSPEC`). *Test:* two browsers with different `max` → node receives the richest folded spec; one reduction fans to both.
- `bridge/adapters.py` dtype-downcast is **subsumed** by node-side reduction; keep the module only if still used elsewhere (grep first).

### Phase 4 — Frontend
- **`thalamus.svelte.ts`** (NEW): per viewer kind, derive the per-axis ViewSpec from the viewer's pixel capacity (`capacity` from element size × devicePixelRatio, hysteresis to avoid renegotiation thrash); fold across `ViewerFeed` consumers of one `(node,slot)`.
- **`api/data.ts`** (MOD): send the seed ViewSpec on connect and `{op:"view"}` on renegotiation up the existing `/data` WS; decode reduced frames unchanged (`$lib/codec/decode`).
- **`ArrayViewer.svelte`**: render the envelope as a min/max **band** keyed off the received `meta.reduced[axis].method==="envelope"` (de-interleave 2×W). *Test (e2e):* Oscillator/PSD slot shows a reduced band that reflows on resize.
- **`MetadataPanel.svelte`**: reconstruct + show the **true original** shape/coords from `meta.reduced` (hide reduction artifacts) — spec §8.9. *Test (e2e):* a reduced slot shows original shape, not `2*W`.
- **Stress/leak (e2e + py):** `test.gfi` with 10+ viewers 60 s ≥ 55 fps; `viewer_count` returns to 0 on abrupt close; no leaked reducer thread after `_reset_for_tests`.

---

## Self-Review (Phase 1)

- **Spec coverage (Phase 1 scope = spec §6.1–6.5):** ViewSpec/AxisSpec/parser (Task 1 ✓ §6.1), `_subsample_idx`/`_envelope`/`_area_axis` (Tasks 2–4 ✓ §6.5), `_set_coord`/`_apply_axis` coord co-reduction + per-axis info (Task 5 ✓ §6.2/6.4), `reduce_for_view` compose/fail-open/meta/Data (Task 6 ✓ §6.2). `_snapshot_for_offer` (§6.6) is deferred to Phase 2 (it runs on the node thread, belongs with `offer`). Per-kind axis lists (§6.3 table) are a frontend concern (Phase 4); the node "applies whatever it receives."
- **Placeholder scan:** none — every code/test step has full bodies and exact commands.
- **Type consistency:** `AxisSpec(axis,max,method)`, `ViewSpec(axes,version)`, `_subsample_idx(n,m)->ndarray`, `_envelope(x,axis,w)->(ndarray,ndarray)`, `_area_axis(x,axis,m)->(ndarray,ndarray)`, `_apply_axis(arr,axis,a,meta)->(ndarray,Optional[dict])`, `_set_coord(meta,dim,new_coord)->None`, `reduce_for_view(data,spec)->Data` — names/signatures consistent across Tasks 1–6 and the Phase-2/3 consumers (`reduce_for_view`, `ViewSpec`, `viewspec_from_dict`).
- **Deviation from spec noted:** `_apply_axis` returns `info=None` for a skipped envelope axis (the spec inlined the skip in prose only); `reduce_for_view` treats `info is None` as "axis untouched" and passthroughs the original `Data` when no axis reduced — this realizes the spec's "envelope skip-if-ratio<2× returns input" acceptance criterion.
