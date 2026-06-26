"""ViewSpec fold contract — the per-axis collision rule (node_reduce.py).

Two reductions targeting the same axis fold to (larger cap, richest method).
That rule lives in ONE kernel (`_fold_axis`) shared by the manager-side spec fold
(`fold_viewspecs`, keyed by raw axis) and the node-side de-dup inside
`reduce_for_view` (keyed by canonical axis) so they can't drift. A committed
golden fixture (tests/viewspec_golden.json) is folded here AND by
capacity.test.ts, pinning the cross-language contract like the codec golden.
"""
import json
from pathlib import Path

from goofi.node_reduce import _fold_axis, fold_viewspecs

GOLDEN = json.loads((Path(__file__).parent / "viewspec_golden.json").read_text(encoding="utf-8"))


# ---- the shared per-axis kernel ----------------------------------------------

def test_fold_axis_takes_the_larger_cap():
    assert _fold_axis(64, "subsample", 256, "subsample") == (256, "subsample")
    assert _fold_axis(256, "subsample", 64, "subsample") == (256, "subsample")


def test_fold_axis_takes_the_richest_method():
    # envelope (3) > area (2) > subsample (1); the peak-preserving-est survives,
    # independent of which side it arrives on.
    assert _fold_axis(10, "subsample", 10, "envelope") == (10, "envelope")
    assert _fold_axis(10, "envelope", 10, "subsample") == (10, "envelope")
    assert _fold_axis(10, "area", 10, "subsample") == (10, "area")


def test_fold_axis_combines_cap_and_method_independently():
    # larger cap from one side, richer method from the other
    assert _fold_axis(100, "subsample", 50, "envelope") == (100, "envelope")


# ---- fold_viewspecs (manager-side, raw-axis keyed) ---------------------------

def test_fold_empty_is_empty():
    assert fold_viewspecs([]) == {"axes": [], "version": 0}


def test_fold_max_of_maxes_and_richest_on_shared_axis():
    out = fold_viewspecs(
        [
            {"axes": [{"axis": -1, "max": 128, "method": "subsample"}], "version": 2},
            {"axes": [{"axis": -1, "max": 64, "method": "envelope"}], "version": 5},
        ]
    )
    assert out == {"axes": [{"axis": -1, "max": 128, "method": "envelope"}], "version": 5}


def test_fold_keeps_raw_axes_separate_and_sorted():
    # axis 0 and axis -1 are NOT canonicalized here (the node does that later); the
    # manager folds by raw axis and emits them sorted ascending.
    out = fold_viewspecs([{"axes": [{"axis": 0, "max": 8, "method": "subsample"}, {"axis": -1, "max": 16, "method": "envelope"}]}])
    assert out == {
        "axes": [
            {"axis": -1, "max": 16, "method": "envelope"},
            {"axis": 0, "max": 8, "method": "subsample"},
        ],
        "version": 0,
    }


# ---- cross-language golden conformance ---------------------------------------

def test_golden_conformance():
    for case in GOLDEN:
        assert fold_viewspecs(case["input"]) == case["output"], case["name"]
