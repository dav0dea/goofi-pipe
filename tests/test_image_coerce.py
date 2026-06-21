"""Shared image dtype coercion helpers (report A0 uint8 convention).

Image producers emit uint8 [0,255]; consumers that need floats call
as_float01(), consumers that feed cv2/PIL call as_uint8(). Both accept either
representation so a node works regardless of which a producer hands it.
"""
import numpy as np

from goofi.image_utils import as_float01, as_uint8


def test_as_uint8_passthrough_for_uint8():
    u = np.array([[0, 127, 255]], dtype=np.uint8)
    out = as_uint8(u)
    assert out.dtype == np.uint8
    assert np.array_equal(out, u)


def test_as_uint8_scales_and_rounds_float01():
    f = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = as_uint8(f)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.array([0, 128, 255], dtype=np.uint8))


def test_as_uint8_clips_out_of_range_floats():
    f = np.array([-0.2, 1.5], dtype=np.float32)
    assert as_uint8(f).tolist() == [0, 255]


def test_as_float01_passthrough_for_float():
    f = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    out = as_float01(f)
    assert out.dtype == np.float32
    assert np.allclose(out, f)


def test_as_float01_scales_uint8():
    u = np.array([0, 51, 255], dtype=np.uint8)
    out = as_float01(u)
    assert out.dtype == np.float32
    assert np.allclose(out, [0.0, 0.2, 1.0], atol=1e-3)


def test_roundtrip_uint8_is_stable():
    u = np.array([0, 1, 127, 200, 255], dtype=np.uint8)
    assert np.array_equal(as_uint8(as_float01(u)), u)
