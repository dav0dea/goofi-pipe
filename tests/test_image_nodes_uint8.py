"""Image nodes under the A0 uint8 convention.

Consumers must produce the same result whether their image input arrives as
float32 [0,1] or uint8 [0,255]; image producers emit uint8. These nodes are
pure cv2/numpy/colorsys, so they run standalone with no transport/hardware.
"""
import numpy as np

from goofi.data import Data, DataType
from goofi.node import NodeEnv


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def _run(node, slot, arr):
    return node(**{slot: Data(DataType.ARRAY, arr, {})})


def _gradient_rgb(h=32, w=32):
    """A smooth float [0,1] RGB image and its uint8 equivalent."""
    base = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    img_f = np.stack([base, base[::-1], base.T % 1.0], axis=-1).astype(np.float32)
    img_u = np.clip(np.round(img_f * 255), 0, 255).astype(np.uint8)
    return img_f, img_u


def _block_rgb(h=32, w=32):
    """Float [0,1] RGB with a hard vertical edge + its uint8 equivalent."""
    img_f = np.zeros((h, w, 3), dtype=np.float32)
    img_f[:, w // 2 :, :] = 1.0
    img_u = (img_f * 255).astype(np.uint8)
    return img_f, img_u


def test_colorenhancer_uint8_matches_float():
    from goofi.nodes.misc.colorenhancer import ColorEnhancer

    f, u = _gradient_rgb()
    out_f = _run(_standalone(ColorEnhancer), "image", f)["enhanced_image"][0]
    out_u = _run(_standalone(ColorEnhancer), "image", u)["enhanced_image"][0]
    assert np.allclose(out_f, out_u, atol=0.02)


def test_rgbtohsv_uint8_matches_float():
    from goofi.nodes.misc.rgbtohsv import RGBtoHSV

    f, u = _gradient_rgb()
    out_f = _run(_standalone(RGBtoHSV), "rgb_image", f)["hsv_image"][0]
    out_u = _run(_standalone(RGBtoHSV), "rgb_image", u)["hsv_image"][0]
    assert np.allclose(out_f, out_u, atol=0.02)


def test_hsvtorgb_uint8_matches_float():
    from goofi.nodes.misc.hsvtorgb import HSVtoRGB

    f, u = _gradient_rgb()
    out_f = _run(_standalone(HSVtoRGB), "hsv_image", f)["rgb_image"][0]
    out_u = _run(_standalone(HSVtoRGB), "hsv_image", u)["rgb_image"][0]
    assert np.allclose(out_f, out_u, atol=0.02)


def test_edgedetector_uint8_no_overflow_and_emits_uint8():
    from goofi.nodes.misc.edgedetector import EdgeDetector

    f, u = _block_rgb()
    out_f = _run(_standalone(EdgeDetector), "image", f)["edges"][0]
    out_u = _run(_standalone(EdgeDetector), "image", u)["edges"][0]
    # Output is a uint8 [0,255] edge map.
    assert out_u.dtype == np.uint8
    # uint8 input must not overflow into garbage — edges agree with the
    # float-input result on the overwhelming majority of pixels.
    assert out_f.shape == out_u.shape
    assert np.mean(out_f != out_u) < 0.02


def test_loadfile_emits_uint8(tmp_path):
    from PIL import Image

    from goofi.nodes.inputs.loadfile import LoadFile

    arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    png = tmp_path / "img.png"
    Image.fromarray(arr).save(png)

    node = _standalone(LoadFile)
    node.setup()
    node.params.file.filename.value = str(png)
    node.params.file.type.value = "image"
    node.load_file()

    assert node.data_output is not None
    assert node.data_output[0].dtype == np.uint8
