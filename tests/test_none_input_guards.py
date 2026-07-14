"""Nodes must tolerate a None input slot (no data delivered yet) by returning
None — not crash.

A constructed `Data` can never carry `data=None` (the constructor rejects a
value that isn't the dtype's concrete type), so `if X.data is None` is both a
dead check AND a crash: when the slot has no data the framework passes `X=None`,
and `X.data` raises AttributeError. The correct guard is `if X is None`. These
seven nodes had it inverted."""
import numpy as np
import pytest

from goofi.data import Data, DataType


def _node(dotted, cls):
    import importlib

    try:
        mod = importlib.import_module(dotted)
    except (ModuleNotFoundError, ImportError) as e:
        pytest.skip(f"{dotted} needs an optional dep unavailable here: {e}")
    return getattr(mod, cls).create_standalone()


def test_stringtotable_none_input_returns_none():
    assert _node("goofi.nodes.misc.stringtotable", "StringToTable").process(None) is None


def test_tabletostring_none_input_returns_none():
    assert _node("goofi.nodes.misc.tabletostring", "TableToString").process(None) is None


def test_binarize_none_input_returns_none():
    assert _node("goofi.nodes.analysis.binarize", "Binarize").process(None) is None


def test_avalanches_none_input_returns_none():
    assert _node("goofi.nodes.analysis.avalanches", "Avalanches").process(None) is None


def test_audiotagging_none_input_returns_none():
    assert _node("goofi.nodes.analysis.audiotagging", "AudioTagging").process(None) is None


def test_img2txt_none_image_returns_none():
    # (prompt, image); the required `image` input being None must not crash.
    assert _node("goofi.nodes.analysis.img2txt", "Img2Txt").process(None, None) is None


def test_audio2txt_none_audio_returns_none():
    # (prompt, audio); the required `audio` input being None must not crash.
    assert _node("goofi.nodes.analysis.audio2txt", "Audio2Txt").process(None, None) is None


def test_audio2txt_present_audio_absent_prompt_does_not_crash(monkeypatch):
    # The optional prompt slot being None (audio present) must not crash on the
    # `prompt.data` deref — it should fall back to an empty prompt.
    node = _node("goofi.nodes.analysis.audio2txt", "Audio2Txt")
    seen = {}

    def fake_generate(prompt, audio):
        seen["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(node, "generate_huggingface", fake_generate)
    node.params.audio_to_text.provider.value = "huggingface"
    audio = Data(DataType.ARRAY, np.zeros(1000, np.float32), {"sfreq": 32000})
    out = node.process(None, audio)
    assert out["generated_text"][0] == "ok"
    assert seen["prompt"] == ""  # None prompt -> empty, not a crash
