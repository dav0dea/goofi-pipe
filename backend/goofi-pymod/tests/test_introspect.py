import json
import os

import pytest

import goofi

HERE = os.path.join(os.path.dirname(__file__), "fixtures")


def introspect(name):
    return json.loads(goofi.introspect(os.path.join(HERE, name)))


def test_psd_like_declarations():
    m = introspect("psd_like.py")
    assert m["doc"] == "Power spectral density."
    assert m["inputs"] == [{"name": "data", "kind": "ARRAY", "trigger": True, "multi": False}]
    assert m["outputs"] == [{"name": "psd", "kind": "ARRAY"}]
    assert m["params"][0] == {
        "group": "welch",
        "name": "nperseg",
        "doc": "Window length in samples.",
        "kind": "int",
        "default": 256,
        "min": 16,
        "max": 4096,
    }
    # An undocumented param omits the key entirely, so the JSON an older node emits is unchanged.
    assert m["params"][1] == {"group": "welch", "name": "average", "kind": "bool", "default": True}
    assert isinstance(m["gil_safe"], bool)


def test_real_import_builds_options():
    import platform

    m = introspect("device_options.py")
    p = m["params"][0]
    assert p["kind"] == "str"
    assert platform.system() in p["options"]


def test_no_node_subclass_raises():
    with pytest.raises(Exception):
        introspect("no_node.py")
