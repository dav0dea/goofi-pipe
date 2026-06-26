"""Unit tests for the node-side thalamus reduction engine (src/goofi/node_reduce.py).

Pure-numpy reduction: fold a Data to a per-axis ViewSpec. FAIL-OPEN, never mutate
the input, fresh contiguous output arrays, coord co-reduction so the reduced Data
satisfies Data.__post_init__'s per-axis coord-length assertion.
"""
import numpy as np

from goofi.data import Data, DataType
from goofi.node_reduce import (
    AxisSpec,
    ViewSpec,
    viewspec_from_dict,
    _subsample_idx,
    _envelope,
    _area_axis,
    _apply_axis,
    reduce_for_view,
)


# ---- Task 1: ViewSpec / parser -------------------------------------------------

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


# ---- Task 2: _subsample_idx ----------------------------------------------------

def test_subsample_idx_picks_spread_indices():
    idx = _subsample_idx(100, 5)
    assert list(idx) == [0, 25, 50, 74, 99]


def test_subsample_idx_caps_at_n_and_is_unique():
    idx = _subsample_idx(3, 10)         # m > n -> at most n, no dups
    assert list(idx) == [0, 1, 2]
    assert len(set(idx)) == len(idx)
    assert list(idx) == sorted(idx)


# ---- Task 3: _envelope ---------------------------------------------------------

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


# ---- Task 4: _area_axis --------------------------------------------------------

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
    ei = np.linspace(0, 7, 4).astype(int)
    ej = np.linspace(0, 5, 3).astype(int)
    ref = np.empty((3, 2), dtype=np.float32)
    for i in range(3):
        for j in range(2):
            ref[i, j] = img[ei[i]:max(ei[i] + 1, ei[i + 1]),
                            ej[j]:max(ej[j] + 1, ej[j + 1])].mean()
    assert np.max(np.abs(a - ref)) < 1e-6


# ---- Task 5: _apply_axis -------------------------------------------------------

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


# ---- Task 6: reduce_for_view ---------------------------------------------------

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
    reduce_for_view(data, viewspec_from_dict(
        {"axes": [{"axis": 0, "max": 256, "method": "envelope"}]}))
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


def test_reduce_for_view_fails_open_on_non_array():
    s = Data(DataType.STRING, "hello", {})
    spec = viewspec_from_dict({"axes": [{"axis": 0, "max": 2, "method": "area"}]})
    assert reduce_for_view(s, spec) is s


def test_reduce_for_view_none_or_empty_spec_passthrough():
    data = Data(DataType.ARRAY, np.zeros((4,), dtype=np.float32), {})
    assert reduce_for_view(data, None) is data
    assert reduce_for_view(data, ViewSpec()) is data


# ---- fold + per-kind default (manager relay folds browser specs) -------------

def test_fold_viewspecs_richest_wins_per_axis():
    from goofi.node_reduce import fold_viewspecs

    folded = fold_viewspecs([
        {"axes": [{"axis": -1, "max": 800, "method": "subsample"}], "version": 1},
        {"axes": [{"axis": -1, "max": 2000, "method": "envelope"}], "version": 4},
    ])
    # max() of max, richest method (envelope>area>subsample), max() of version.
    assert folded["axes"] == [{"axis": -1, "max": 2000, "method": "envelope"}]
    assert folded["version"] == 4


def test_fold_viewspecs_multiple_axes_independent():
    from goofi.node_reduce import fold_viewspecs

    folded = fold_viewspecs([
        {"axes": [{"axis": 0, "max": 4, "method": "subsample"},
                  {"axis": 1, "max": 500, "method": "envelope"}]},
        {"axes": [{"axis": 0, "max": 8, "method": "subsample"},
                  {"axis": 1, "max": 1200, "method": "envelope"}]},
    ])
    assert folded["axes"] == [
        {"axis": 0, "max": 8, "method": "subsample"},
        {"axis": 1, "max": 1200, "method": "envelope"},
    ]


def test_fold_viewspecs_empty_is_no_reduction():
    from goofi.node_reduce import fold_viewspecs

    assert fold_viewspecs([]) == {"axes": [], "version": 0}
    assert fold_viewspecs([{"axes": []}, {"axes": []}]) == {"axes": [], "version": 0}


def test_default_viewspec_for_kind():
    from goofi.node_reduce import default_viewspec_for_kind, viewspec_from_dict

    line = default_viewspec_for_kind("line")
    assert line["axes"][0]["method"] == "envelope"
    # round-trips through the parser the node uses
    assert viewspec_from_dict(line).axes[0].method == "envelope"
    # unknown / non-reducible kinds -> no reduction
    assert default_viewspec_for_kind("string") == {"axes": [], "version": 0}
    assert default_viewspec_for_kind("topomap") == {"axes": [], "version": 0}


# ---- image area reduction preserves aspect (no per-axis distortion) ----------

def test_area_reduction_preserves_aspect_square():
    # 512x512 square source, NON-square viewer box (128 tall, 256 wide) -> the
    # reduced image must stay square (uniform downscale), not 128x256.
    img = Data(DataType.ARRAY, np.zeros((512, 512, 3), dtype=np.float32), {})
    spec = viewspec_from_dict({"axes": [
        {"axis": 0, "max": 128, "method": "area"},
        {"axis": 1, "max": 256, "method": "area"},
    ]})
    red = reduce_for_view(img, spec)
    assert red.data.shape[0] == red.data.shape[1]  # aspect preserved (square stays square)
    assert red.data.shape == (128, 128, 3)          # min scale = 128/512 applied to both


def test_area_reduction_preserves_aspect_wide():
    # 1080x1920 (16:9) -> 720x1280 box: equal scale on both -> exact 16:9 kept.
    img = Data(DataType.ARRAY, np.zeros((1080, 1920, 3), dtype=np.float32), {})
    spec = viewspec_from_dict({"axes": [
        {"axis": 0, "max": 720, "method": "area"},
        {"axis": 1, "max": 1280, "method": "area"},
    ]})
    red = reduce_for_view(img, spec)
    assert red.data.shape == (720, 1280, 3)


def test_area_passthrough_when_image_already_fits():
    # An image smaller than the viewer box must pass through verbatim: no area
    # downscale (the identity block-mean was a wasted full copy) and no 'reduced'
    # marker, so reduce_for_view returns the original Data object untouched.
    img = Data(DataType.ARRAY, np.arange(8 * 8, dtype=np.float32).reshape(8, 8), {})
    spec = viewspec_from_dict({"axes": [
        {"axis": 0, "max": 64, "method": "area"},
        {"axis": 1, "max": 64, "method": "area"},
    ]})
    red = reduce_for_view(img, spec)
    assert red is img  # fail-open passthrough — nothing actually reduced


# ---- HIGH bug regression: 1-D line must envelope (peaks), not subsample --------

def test_folded_line_spec_envelopes_1d_not_subsample():
    """The line viewer sends a channel-subsample (axis 0) AND a sample-envelope
    (axis -1). After the manager fold sorts axes ascending, BOTH collapse onto
    axis 0 for a 1-D buffer; the node must keep the RICHER (envelope) so a
    single-channel waveform preserves its peaks instead of being aliased-subsampled."""
    from goofi.node_reduce import fold_viewspecs

    line = {"axes": [
        {"axis": 0, "max": 300, "method": "subsample"},
        {"axis": -1, "max": 1600, "method": "envelope"},
    ], "version": 1}
    folded = fold_viewspecs([line])               # exactly what the manager sends the node
    spec = viewspec_from_dict(folded)
    n = 10000
    data = Data(DataType.ARRAY, np.linspace(-1, 1, n).astype(np.float32), {})
    red = reduce_for_view(data, spec)
    assert red.meta["reduced"]["0"]["method"] == "envelope"   # NOT 'subsample'
    assert red.data.shape == (3200,)                          # 2*W envelope, not (300,) subsample
