"""Viewer-adapter unit tests (viewer-adapters-design 2026-06-21, backlog #3).

Adapters convert a node's pure-float `Data` into the representation a given
viewer kind needs (uint8 image / float16 line / passthrough string-table) at the
bridge boundary, attaching float range/stats under the namespaced `meta["__view__"]`
key so range and the metadata inspector stay float-accurate. These tests pin the
per-kind conversion contract; the data-plane wiring is tested separately.
"""
import numpy as np
import pytest

from goofi.bridge.adapters import ADAPTERS, adapt, view_stats
from goofi.data import Data, DataType


def _arr(a, **meta):
    return Data(DataType.ARRAY, np.asarray(a), dict(meta))


# --- view_stats: float stats helper ----------------------------------------

def test_view_stats_computes_float_min_mean_max():
    s = view_stats(np.array([0.0, 2.0, 4.0], dtype=np.float32))
    assert s["min"] == pytest.approx(0.0)
    assert s["mean"] == pytest.approx(2.0)
    assert s["max"] == pytest.approx(4.0)


# --- image: RGB clamp (no normalization, preserves colour) ------------------

def test_image_rgb_clamps_without_normalizing():
    img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)  # (1,1,3) RGB
    out = adapt(_arr(img), "image")
    assert out.data.dtype == np.uint8
    # 0->0, 0.5->128, 1.0->255 : a straight [0,1]->[0,255] clamp, not stretched
    assert out.data.ravel().tolist() == [0, 128, 255]
    assert out.meta["__view__"]["range"] == [pytest.approx(0.0), pytest.approx(1.0)]
    assert out.meta["__view__"]["stats"]["max"] == pytest.approx(1.0)


# --- image: grayscale normalize [fmin,fmax]->[0,255] ------------------------

def test_image_grayscale_normalizes_to_full_range():
    g = np.array([[0.2, 0.45], [0.7, 0.95]], dtype=np.float32)  # ndim==2
    out = adapt(_arr(g), "image")
    assert out.data.dtype == np.uint8
    # the data min maps to 0 and the data max maps to 255 (full colormap range)
    assert int(out.data[g == g.min()][0]) == 0
    assert int(out.data[g == g.max()][0]) == 255
    assert out.meta["__view__"]["range"] == [pytest.approx(0.2), pytest.approx(0.95)]


def test_image_grayscale_flat_image_does_not_blow_up():
    flat = np.full((2, 2), 0.7, dtype=np.float32)  # fmax == fmin
    out = adapt(_arr(flat), "image")
    assert out.data.dtype == np.uint8
    assert np.isfinite(out.data).all()  # epsilon guard, no NaN/inf


# --- line / trajectory / topomap: float16 with exact float32 stats ----------

def test_line_downcasts_to_float16_with_exact_stats():
    line = np.linspace(-5.0, 5.0, 64, dtype=np.float32).reshape(2, 32)
    out = adapt(_arr(line), "line")
    assert out.data.dtype == np.float16
    # stats are computed on the float32 array (pre-downcast), so they are exact
    assert out.meta["__view__"]["stats"]["min"] == pytest.approx(-5.0)
    assert out.meta["__view__"]["stats"]["max"] == pytest.approx(5.0)
    assert "range" not in out.meta["__view__"]  # range is image-only


def test_trajectory_and_topomap_also_float16():
    traj = adapt(_arr(np.zeros((2, 10), dtype=np.float32)), "trajectory")
    topo = adapt(_arr(np.zeros((8,), dtype=np.float32)), "topomap")
    assert traj.data.dtype == np.float16
    assert topo.data.dtype == np.float16


# --- string / table: passthrough, no __view__ -------------------------------

def test_string_passthrough_unchanged():
    s = Data(DataType.STRING, "hello", {})
    out = adapt(s, "string")
    assert out.dtype == DataType.STRING and out.data == "hello"
    assert "__view__" not in out.meta


def test_table_passthrough_unchanged():
    t = Data(DataType.TABLE, {"a": _arr(np.zeros(3, dtype=np.float32))}, {})
    out = adapt(t, "table")
    assert out.dtype == DataType.TABLE
    assert "__view__" not in out.meta


# --- non-renderable (ndim>3): summary frame, no array body ------------------

def test_non_renderable_emits_summary_no_body():
    big = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)  # ndim 4
    out = adapt(_arr(big), "image")
    summary = out.meta["__view__"]["summary"]
    assert list(summary["shape"]) == [2, 2, 2, 2]
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(15.0)
    assert out.data.size == 0  # no heavy array body on the wire


# --- raw / unknown fallback -------------------------------------------------

def test_raw_kind_returns_data_unchanged():
    d = _arr(np.zeros(3, dtype=np.float32))
    assert adapt(d, "raw") is d


def test_unknown_kind_falls_back_to_raw():
    d = _arr(np.zeros(3, dtype=np.float32))
    assert adapt(d, "nonsense") is d


def test_registry_exposes_every_viewer_kind():
    for kind in ("image", "line", "trajectory", "topomap", "string", "table"):
        assert kind in ADAPTERS


# --- non-aliasing: adapting must not mutate the input meta ------------------

def test_adapt_does_not_mutate_input_meta():
    d = _arr(np.zeros((4, 4, 3), dtype=np.float32), channels={})
    before = dict(d.meta)
    adapt(d, "image")
    assert "__view__" not in d.meta  # output __view__ never leaks back to the node Data
    assert d.meta.keys() == before.keys()
