"""Join must combine metadata sensibly and guard shape compatibility.

join.py historically (a) took sfreq only from input `a`, silently dropping b's
when a had none, and (b) ran concatenate/stack with no shape check, so an
incompatible pair surfaced as a cryptic numpy error instead of one that names the
offending shapes. These tests pin the fixed behavior. They also assert the
producers' meta is untouched, so the deepcopy(a.meta) guard against fan-out
aliasing is not regressed by the fix.
"""
from copy import deepcopy

import numpy as np
import pytest

from goofi.data import Data, DataType
from goofi.node import NodeEnv
from goofi.nodes.array.join import Join


def _standalone(cls):
    i, o, p = cls._configure()
    return cls(None, i, o, p, NodeEnv.STANDALONE)


def _concat_node(axis=0):
    node = _standalone(Join)
    node.params.join.method.value = "concatenate"
    node.params.join.axis.value = axis
    return node


def _stack_node(axis=0):
    node = _standalone(Join)
    node.params.join.method.value = "stack"
    node.params.join.axis.value = axis
    return node


def test_concatenate_carries_b_sfreq_when_a_lacks_it():
    # a has no sfreq; b's must be carried through rather than dropped.
    node = _concat_node()
    a = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    b = Data(DataType.ARRAY, np.zeros((3, 3), np.float32), {"sfreq": 128.0})
    _, meta = node(a=a, b=b)["out"]
    assert meta.get("sfreq") == 128.0


def test_concatenate_keeps_a_sfreq_when_both_present():
    # When both sides carry sfreq they should match; a is authoritative.
    node = _concat_node()
    a = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {"sfreq": 128.0})
    b = Data(DataType.ARRAY, np.zeros((3, 3), np.float32), {"sfreq": 256.0})
    _, meta = node(a=a, b=b)["out"]
    assert meta["sfreq"] == 128.0


def test_concatenate_without_sfreq_does_not_crash():
    # Absent on both sides: leave sfreq unset, never raise.
    node = _concat_node()
    a = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    b = Data(DataType.ARRAY, np.zeros((3, 3), np.float32), {})
    _, meta = node(a=a, b=b)["out"]
    assert "sfreq" not in meta


def test_concatenate_incompatible_nonjoin_axis_raises_clear_error():
    # Non-join axis differs (3 != 4): the error must name both shapes.
    node = _concat_node(axis=0)
    a = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    b = Data(DataType.ARRAY, np.zeros((2, 4), np.float32), {})
    with pytest.raises(ValueError) as excinfo:
        node(a=a, b=b)
    msg = str(excinfo.value)
    assert "(2, 3)" in msg and "(2, 4)" in msg


def test_stack_incompatible_shapes_raises_clear_error():
    # Stack requires identical shapes; the error must name both shapes.
    node = _stack_node(axis=0)
    a = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    b = Data(DataType.ARRAY, np.zeros((2, 4), np.float32), {})
    with pytest.raises(ValueError) as excinfo:
        node(a=a, b=b)
    msg = str(excinfo.value)
    assert "(2, 3)" in msg and "(2, 4)" in msg


def test_compatible_concatenate_and_stack_still_work():
    cnode = _concat_node(axis=0)
    a = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    b = Data(DataType.ARRAY, np.zeros((3, 3), np.float32), {})
    val, _ = cnode(a=a, b=b)["out"]
    assert val.shape == (5, 3)

    snode = _stack_node(axis=0)
    a2 = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    b2 = Data(DataType.ARRAY, np.zeros((2, 3), np.float32), {})
    val2, _ = snode(a=a2, b=b2)["out"]
    assert val2.shape == (2, 2, 3)


def test_join_does_not_mutate_producer_meta():
    # The deepcopy(a.meta) guard must keep both producers' meta intact.
    node = _concat_node(axis=0)
    a = Data(
        DataType.ARRAY,
        np.zeros((2, 3), np.float32),
        {"sfreq": 128.0, "channels": {"dim0": ["a0", "a1"]}},
    )
    b = Data(
        DataType.ARRAY,
        np.zeros((3, 3), np.float32),
        {"sfreq": 256.0, "channels": {"dim0": ["b0", "b1", "b2"]}},
    )
    a_before = deepcopy(a.meta)
    b_before = deepcopy(b.meta)
    node(a=a, b=b)
    assert a.meta == a_before, "Join mutated producer a's meta"
    assert b.meta == b_before, "Join mutated producer b's meta"
