"""Switch must pair each selected input's data with that SAME input's meta.

In string mode the selector==3 branch had a copy-paste typo: it returned
string3.data with string2.meta. When string2 is unwired (None) and selector=3,
reading string2.meta raised AttributeError every tick; when string2 was wired the
output silently carried the wrong metadata. These tests pin self-paired meta.
"""
import numpy as np

from goofi.data import Data, DataType
from goofi.node import NodeEnv
from goofi.nodes.misc.switch import Switch


def _standalone(cls):
    i, o, p = cls._configure()
    return cls(None, i, o, p, NodeEnv.STANDALONE)


def _string_switch(selector_value):
    node = _standalone(Switch)
    node.params.settings.mode.value = "string"
    return node, Data(DataType.ARRAY, np.array([float(selector_value)]), {})


def test_string_selector3_with_string2_unwired_does_not_raise():
    # The reproducer: string2 unwired (None), selector=3, string3 wired. The
    # buggy branch dereferenced string2.meta -> AttributeError on None every tick.
    node, selector = _string_switch(3)
    string3 = Data(DataType.STRING, "hello", {"k": "v"})
    # process() directly so an unwired input can be passed as None (the node(...)
    # harness's to_data() would reject an explicit None).
    out = node.process(
        selector=selector,
        array1=None,
        array2=None,
        array3=None,
        string1=None,
        string2=None,
        string3=string3,
    )
    _, meta = out["string_out"]
    # Data injects a `dtype` key into meta, so compare the discriminating key.
    assert meta["k"] == "v", "selector=3 must carry string3's own meta"


def test_string_selector3_carries_string3_meta_not_string2():
    # Even when string2 IS wired, selector=3 must forward string3's meta, not
    # string2's. Distinct metas make the mispairing observable.
    node, selector = _string_switch(3)
    string2 = Data(DataType.STRING, "two", {"k": "two"})
    string3 = Data(DataType.STRING, "three", {"k": "three"})
    out = node.process(
        selector=selector,
        array1=None,
        array2=None,
        array3=None,
        string1=None,
        string2=string2,
        string3=string3,
    )
    value, meta = out["string_out"]
    assert value == "three"
    assert meta["k"] == "three", "selector=3 mispaired string3.data with string2.meta"
