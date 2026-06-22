"""Image nodes under the pure-float convention (viewer-adapters revert of A0).

Nodes process pure float again: producers emit float [0,1], consumers accept and
emit float, and any uint8 a C library needs (cv2/mediapipe in edgedetector/
poseestimation) is coerced *internally* and never reaches an output slot. The
uint8-on-the-wire bandwidth win now lives in the bridge viewer adapters, not the
nodes. These nodes are pure cv2/numpy/colorsys, so they run standalone.
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
    base = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    return np.stack([base, base[::-1], base.T % 1.0], axis=-1).astype(np.float32)


def _block_rgb(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, w // 2 :, :] = 1.0
    return img


# --- producers emit float ---------------------------------------------------

def test_loadfile_emits_float(tmp_path):
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
    out = node.data_output[0]
    assert out.dtype.kind == "f"          # float, not uint8
    assert out.max() <= 1.0 + 1e-6        # normalized to [0,1]


# --- edgedetector emits a float [0,1] edge map (uint8 only internal to cv2) --

def test_edgedetector_emits_float_edge_map():
    from goofi.nodes.misc.edgedetector import EdgeDetector

    out = _run(_standalone(EdgeDetector), "image", _block_rgb())["edges"][0]
    assert out.dtype.kind == "f"
    assert out.max() <= 1.0 + 1e-6


# --- consumers accept float and emit float (no uint8 coercion at the slot) ---

def test_colorenhancer_emits_float():
    from goofi.nodes.misc.colorenhancer import ColorEnhancer

    out = _run(_standalone(ColorEnhancer), "image", _gradient_rgb())["enhanced_image"][0]
    assert out.dtype.kind == "f"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rgbtohsv_emits_float():
    from goofi.nodes.misc.rgbtohsv import RGBtoHSV

    out = _run(_standalone(RGBtoHSV), "rgb_image", _gradient_rgb())["hsv_image"][0]
    assert out.dtype.kind == "f"


def test_hsvtorgb_emits_float():
    from goofi.nodes.misc.hsvtorgb import HSVtoRGB

    out = _run(_standalone(HSVtoRGB), "hsv_image", _gradient_rgb())["rgb_image"][0]
    assert out.dtype.kind == "f"


# --- the node files no longer import the bridge-only coercion helpers --------

def test_nodes_do_not_import_image_utils():
    import goofi.nodes.inputs.loadfile as loadfile
    import goofi.nodes.inputs.videostream as videostream
    import goofi.nodes.misc.colorenhancer as colorenhancer
    import goofi.nodes.misc.edgedetector as edgedetector

    for mod in (loadfile, videostream, colorenhancer, edgedetector):
        assert not hasattr(mod, "as_uint8"), f"{mod.__name__} still imports as_uint8"
        assert not hasattr(mod, "as_float01"), f"{mod.__name__} still imports as_float01"
