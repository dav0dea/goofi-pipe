"""Characterization of the Constant{String,Array} overwrite-hold state machine.

These two nodes hold the latest `overwrite` input for `overwrite_timeout` seconds, then
revert to their constant. The two differ ONLY in the held-branch meta: ConstantString emits
{}, ConstantArray carries the override's OWN meta. These tests pin that (esp. the array
meta) so the shared-mixin extraction is provably behavior-preserving.
"""
import time

import numpy as np

from goofi.data import Data, DataType
from goofi.node import NodeEnv
from goofi.nodes.inputs.constantarray import ConstantArray
from goofi.nodes.inputs.constantstring import ConstantString


def _standalone(cls):
    i, o, p = cls._configure()
    n = cls(None, i, o, p, NodeEnv.STANDALONE)
    n.setup()  # STANDALONE skips auto-setup; init the overwrite state
    return n


def test_constantstring_returns_constant_without_overwrite():
    n = _standalone(ConstantString)
    n.params.constant.value.value = "hello"
    val, meta = n.process(None)["out"]
    assert val == "hello"
    assert meta == {}


def test_constantstring_holds_then_expires_overwrite():
    n = _standalone(ConstantString)
    n.params.constant.value.value = "base"
    n.params.constant.overwrite_timeout.value = 5

    val, meta = n.process(Data(DataType.STRING, "ovr", {}))["out"]
    assert val == "ovr" and meta == {}  # override held, empty meta
    assert n.process(None)["out"][0] == "ovr"  # still held next tick

    n.last_overwrite_time = time.time() - 100  # force the timeout to elapse
    assert n.process(None)["out"][0] == "base"  # reverted to the constant


def test_constantstring_zero_timeout_never_expires():
    n = _standalone(ConstantString)
    n.params.constant.value.value = "base"
    n.params.constant.overwrite_timeout.value = 0
    n.process(Data(DataType.STRING, "ovr", {}))
    n.last_overwrite_time = time.time() - 10_000
    assert n.process(None)["out"][0] == "ovr"  # 0 never clears the override


def test_constantarray_returns_ones_with_sfreq_without_overwrite():
    n = _standalone(ConstantArray)
    n.params.constant.value.value = 2.0
    n.params.constant.shape.value = "3"
    n.params.constant.graph.value = "none"
    val, meta = n.process(None)["out"]
    assert np.allclose(val, np.ones(3) * 2.0)
    assert "sfreq" in meta


def test_constantarray_overwrite_carries_its_own_meta_then_expires():
    n = _standalone(ConstantArray)
    n.params.constant.shape.value = "3"
    n.params.constant.graph.value = "none"
    n.params.constant.overwrite_timeout.value = 5

    ovr = Data(DataType.ARRAY, np.array([9.0, 8.0]), {"channels": {"dim0": ["x", "y"]}})
    val, meta = n.process(ovr)["out"]
    assert np.allclose(val, [9.0, 8.0])
    # The held branch carries the override's OWN meta (channels), NOT the default's sfreq.
    assert meta.get("channels") == {"dim0": ["x", "y"]}
    assert "sfreq" not in meta

    n.last_overwrite_time = time.time() - 100  # expire → ones default with sfreq
    val, meta = n.process(None)["out"]
    assert val.shape == (3,)
    assert "sfreq" in meta and "channels" not in meta
