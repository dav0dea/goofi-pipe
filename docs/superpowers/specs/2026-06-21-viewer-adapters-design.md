# Viewer adapters — backend per-kind representation for the data plane

**Status:** design approved, pending implementation plan
**Date:** 2026-06-21

## Problem

The A0 work standardized image producers on `uint8` and made consumers coerce,
so that the wire carried uint8 (4× smaller than float32) and the bridge could
forward producer bytes verbatim. This was a **misconfiguration**: goofi nodes
are floating-point processors — an image flowing through edge-detect, colour,
HSV, pose, etc. must stay float for the maths to be correct. Forcing uint8 at
the producer corrupts that pipeline and leaks a lossy representation into places
the user should never see it (node processing, the metadata inspector, the
min/mean/max stats shown for odd-shaped arrays).

The bandwidth win is still worth having — but it belongs at the **viewer
boundary**, not in the nodes. The frontend already knows what each viewer wants
(an image viewer wants pixels; a line viewer wants samples), and those wants are
narrower than the full-precision float the nodes compute. So the conversion
should happen per-viewer, on the way out, and stay invisible everywhere else.

## Goals

- Nodes process **pure float** again. No node emits or coerces uint8. (Revert A0.)
- A **per-viewer-kind adapter** in the bridge converts the node's float `Data`
  into the representation that kind needs, with a lower-bit transport where it
  loses nothing the viewer can show:
  - image → `uint8`
  - line / trajectory / topomap → `float16`
  - string → string (unchanged)
  - table → pass-through (recursive per-entry adapting is a future extension)
- The conversion is **fully invisible** outside the viewer's own rendering:
  - the **viewer value range** follows the original **float** data range, not the
    uint8/float16 range;
  - the **min / mean / max stats** shown for non-renderable shapes are computed
    on the **float** data, not the converted data.
- Bandwidth: HD RGB video stays ~4× smaller on the wire (uint8); line/PSD/EEG
  streams roughly halve (float16) with no visible loss.

## Non-goals (this cut)

- A user-facing precision setting. Precision is fixed by viewer kind. (A cog-menu
  control can be added later; the settings schema already supports it.)
- Recursive per-entry adapting of TABLE data. Tables pass through as-is — they
  carry small structural metadata, not heavy image/array payloads.
- A reconfigure-in-place protocol on the data WS. Switching viewer kind
  re-subscribes (see Data plane), which is a ~1-frame gap on a manual switch.

## Architecture

```
node (float Data) ──iceoryx2──► NodeRef data pump ──► bridge DataHub
                                                          │  decode ONCE per (uid,slot)
                                                          │  for each distinct kind subscribed:
                                                          │     adapt(float Data) → view Data
                                                          │     encode → bytes (memoized per kind/frame)
                                                          ▼
   browser ◄────── WS /data/<node>/<slot>/<kind> (uint8 | float16 | string … + meta.__view__)
                                                          │
                                                       viewer: renders the converted array,
                                                       reads range/stats from meta.__view__ (float)
```

The node→bridge path and the iceoryx2 transport are unchanged. The adapter layer
lives entirely in the bridge (server-side, viewer-only). uint8/float16 never
re-enter node processing, persistence, or the metadata inspector.

### Adapter layer — `src/goofi/bridge/adapters.py` (new)

- An adapter is a pure function `adapt(data: Data) -> Data`, registered by kind.
- Registry: `ADAPTERS = {"image": …, "line": …, "trajectory": …, "topomap": …,
  "string": …, "table": …}` plus a `"raw"` fallback (returns `data` unchanged)
  for unknown kinds.
- Stats helper computes `{min, mean, max}` on the **float** array once and is
  reused by the adapters and the non-renderable summary.

Per-kind behaviour:

- **image** — output `uint8` array + `meta["__view__"]["range"] = [fmin, fmax]`
  and `["stats"] = {min, mean, max}` from the float array.
  - RGB / RGBA (`ndim==3`, channels 3/4): clamp `[0,1] → [0,255]` (no
    normalization — preserves colour).
  - Grayscale (`ndim==2`, or `ndim==3` channel 1): normalize `[fmin,fmax] →
    [0,255]` so the colormap uses the full range; the frontend reconstructs the
    float value as `fmin + (u/255)*(fmax-fmin)` for the colormap + any manual
    window.
- **line / trajectory / topomap** — downcast the array to `float16`; attach
  `meta["__view__"]["stats"]` computed on the original **float32** array (before
  downcast, so stats are exact).
- **string** — unchanged (`raw` passthrough; STRING bodies are already compact).
- **table** — passthrough this cut.
- **Non-renderable** (any kind, when the array shape can't be drawn — e.g.
  `ndim > 3`): emit a **summary** frame — no array body, just
  `meta["__view__"]["summary"] = {shape, dtype, min, mean, max}` from the float
  data. The frontend's high-dim fallback renders text from this.

`image_utils.as_uint8` / `as_float01` move out of the nodes and into the adapter
layer (their only legitimate use is here, server-side).

### Wire contract

The GOOF codec (`codec.py`) is unchanged. The adapter's output `Data` is encoded
normally. The single convention added is a namespaced meta key so node meta is
never clobbered:

```python
meta["__view__"] = {
    "range":   [fmin, fmax],                       # image only
    "stats":   {"min": …, "mean": …, "max": …},    # array kinds
    "summary": {"shape": [...], "dtype": "<f4",     # non-renderable only
                "min": …, "mean": …, "max": …},
}
```

Wire array dtype is just the array's dtype tag: `|u1` (image), `<f2`
(line/trajectory/topomap), node-native for `raw`/string/table.

`meta["__view__"]` is present only for the array/image kinds; string, table, and
`raw` frames omit it, so the frontend reads it defensively (absent → fall back to
computing range/stats from the received array, as today). Grayscale normalization
guards a flat image (`fmax == fmin`) with a small epsilon span.

### Data plane — `src/goofi/bridge/data.py`

- Route gains the kind: `GET /data/<node>/<slot>/<kind>`. (`<kind>` ∈ the
  `ViewerKind` vocabulary plus `raw`.)
- `_SlotMux` still owns **one** iceoryx2 subscription per `(uid, slot)`, but now
  registers `ref.set_data_handler(slot, on_frame, raw=False)` — `on_frame`
  receives the **decoded float `Data`**.
- Forwarders are grouped by kind. Each frame:
  1. for each **distinct kind** among current forwarders, compute
     `bytes_k = encode_data(ADAPTERS[kind](data))` **once** (memoized per frame);
  2. dispatch `bytes_k` to every forwarder of that kind.
  This keeps decode at once-per-slot and adapt at once-per-(slot,kind)-per-frame
  regardless of how many viewers are attached.
- Latest-wins backpressure per forwarder is unchanged.
- This is the deliberate reversal of A1's verbatim forward: the bridge now
  decodes and re-encodes. Cost is one decode + N_kinds adapt/encode per frame per
  viewed slot; the adapters are numpy-vectorized.

### Frontend

- **Codec** (`$lib/codec/decode`): add `<f2` handling — read the half-float body
  with a `DataView` and upcast to `Float32Array` (do not depend on
  `Float16Array` browser support). uint8 already decodes.
- **Worker / data.ts**: thread `kind` through `subscribeData(node, slot, kind,
  cb)`, the worker `sub`/`unsub` messages, the WS URL, and the `(node, slot,
  kind)` ref-count key.
- **Viewers**:
  - `ImageViewer` renders the uint8 array; reads `meta.__view__.range` for the
    colormap window and reconstructs float for a manual window.
  - `HighDimFallback` renders from `meta.__view__.summary`.
  - `ArrayViewer` / `TrajectoryViewer` / `TopomapViewer` consume the
    float16→float32 array and use `meta.__view__.stats` for auto-range.
- **Kind resolution at subscribe time**: the resolved kind (`resolveKind`) is
  known from the stored kind + the slot dtype. If an arriving frame's dtype
  forces a different kind (STRING/TABLE), the viewer re-resolves and
  re-subscribes.

### A0 revert (part of this work)

- Revert commits `74e7fd3` (node uint8 convention) and `834b7de` (image_utils as
  a node dependency) and the `f7a84c7` img2img compensation: producers
  (videostream, loadfile, imagegeneration, edgedetector) emit float again;
  consumers (colorenhancer, hsvtorgb, rgbtohsv, poseestimation) drop their
  coercion. Nodes that genuinely need uint8 for a C library (cv2/mediapipe in
  edgedetector/poseestimation) coerce **internally** as they did before A0 —
  that is the node's own concern and never changes its output dtype.
- `tests/test_image_nodes_uint8.py` is replaced by a test asserting these nodes
  emit **float** images.
- `image_utils.py` is retained but its consumers become the adapter layer, not
  nodes.

## Testing

- **Backend unit** (`tests/test_view_adapters.py`): each kind — float `Data` in →
  expected output dtype + `meta.__view__` range/stats; grayscale vs RGB image
  conversion; non-renderable → summary (no array body); stats match float.
- **Backend nodes**: invert the A0 test — image producers/consumers emit float.
- **Data plane** (`tests/` or e2e): subscribe `…/<kind>`, assert received dtype
  (uint8 for image, float16 for line) and `meta.__view__` float range/stats; two
  viewers of different kinds on one slot each get their own representation.
- **Frontend unit**: `<f2` decode → correct Float32Array; ImageViewer reads
  `meta.__view__.range`; HighDimFallback reads `summary`.
- **e2e**: image viewer shows the float range (not 0–255); a line stream renders
  from a float16 frame; high-dim stats reflect float values.

## Risks / open points

- **Per-frame transcode cost.** The bridge now decodes + adapts + encodes each
  viewed frame (vs verbatim forward). Acceptable: decode is once-per-slot, adapt
  is numpy-vectorized and once-per-kind, and only *viewed* slots pay it. Watch
  the HD-video stress patch for regressions vs the A0 raw-forward baseline.
- **Grayscale 256-level quantization.** Normalizing grayscale to uint8 quantizes
  to 256 levels before the colormap; a manual window much narrower than the data
  range loses precision. Acceptable for visualization; float16 grayscale is a
  fallback option if it bites.
- **Initial kind unknown.** If the slot dtype isn't known before the first frame,
  the first subscription may use the default kind and re-subscribe once the dtype
  arrives (STRING/TABLE). One-frame settle on viewer mount.
