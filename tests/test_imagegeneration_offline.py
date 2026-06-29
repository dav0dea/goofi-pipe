"""ImageGeneration must load Stable Diffusion offline-first.

The two `from_pretrained()` calls used to download from the internet
unconditionally, so `setup()` failed with no network access even when the model
was already cached locally. We mirror the embedding node's offline-tolerant
load: try `local_files_only=True` first (serves the local cache, raises OSError
when the model isn't cached), then fall back to the normal, downloading load.

Loading real Stable Diffusion is heavy and needs a GPU + a multi-GB download, so
we never touch `diffusers` here. Instead we inject fake pipeline classes whose
`from_pretrained` raises OSError for the offline attempt and returns a sentinel
for the fallback, then drive the load helper directly and assert the order.
"""
import types

from goofi.node import NodeEnv
from goofi.nodes.inputs.imagegeneration import ImageGeneration

SENTINEL = "loaded-pipe"


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


class _FakePipeline:
    """Stand-in for a diffusers pipeline class.

    Records the kwargs of every `from_pretrained` call so the test can verify
    the offline-first ordering. The offline attempt (``local_files_only=True``)
    raises OSError just like diffusers does on a cache miss; the fallback returns
    a sentinel object.
    """

    def __init__(self):
        self.calls = []

    def from_pretrained(self, model_id, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError(f"{model_id} is not cached locally")
        return SENTINEL


def _make_node(enabled):
    node = _standalone(ImageGeneration)
    node.params.image_generation.model_id.value = "stabilityai/stable-diffusion-2-1"
    node.params.img2img.enabled.value = enabled
    # Inject fakes for the lazily-imported libs the load helper reaches through.
    node.torch = types.SimpleNamespace(float16="float16")
    node.diffusers = types.SimpleNamespace(
        StableDiffusionPipeline=_FakePipeline(),
        StableDiffusionImg2ImgPipeline=_FakePipeline(),
    )
    return node


def test_text2img_load_prefers_cache_then_falls_back():
    node = _make_node(enabled=False)
    pipe_cls = node.diffusers.StableDiffusionPipeline

    result = node._load_sd_pipe()

    assert result == SENTINEL, "load should succeed via the downloading fallback"
    assert len(pipe_cls.calls) == 2, "should try offline first, then fall back"
    assert pipe_cls.calls[0].get("local_files_only") is True, "first attempt must be offline (cache only)"
    assert not pipe_cls.calls[1].get("local_files_only"), "fallback must allow downloading"


def test_img2img_load_prefers_cache_then_falls_back():
    node = _make_node(enabled=True)
    pipe_cls = node.diffusers.StableDiffusionImg2ImgPipeline

    result = node._load_sd_pipe()

    assert result == SENTINEL, "load should succeed via the downloading fallback"
    assert len(pipe_cls.calls) == 2, "should try offline first, then fall back"
    assert pipe_cls.calls[0].get("local_files_only") is True, "first attempt must be offline (cache only)"
    assert not pipe_cls.calls[1].get("local_files_only"), "fallback must allow downloading"
