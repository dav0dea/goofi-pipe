"""PhiID one-vs-others mode must validate `tgt_index` against the real channel count.

`tgt_index` is an unconstrained IntParam (range [-1, 10]) that has no knowledge of the
actual number of input channels. In one-vs-others mode the node does `tgt = data[tgt_index]`,
so an out-of-bounds index either mis-indexes via numpy negative-wraparound or raises a
cryptic IndexError deep inside `process`. One-vs-others also only makes sense with at
least 2 channels (one target + at least one "other"). These tests pin down a clear,
early ValueError for both failure modes, and guard that a valid index still passes the
bounds check.

phyid is an optional dependency that is not installed here, so we mark setup as done and
stub `calc_PhiID` — the validation we care about must fire *before* any calc_PhiID call.
"""
from collections import defaultdict

import numpy as np
import pytest

from goofi.data import Data, DataType
from goofi.node import NodeEnv
from goofi.nodes.analysis.phiid import PhiID


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def _make_node(mode="one-vs-others", tgt_index=0):
    node = _standalone(PhiID)
    # phyid isn't installed; skip setup() (which imports it) and stub the calculator so
    # process() can run far enough to exercise (or pass) the bounds check.
    node._setup_done.set()
    node.calc_PhiID = lambda src, tgt, tau, kind="gaussian", redundancy="MMI": (
        defaultdict(lambda: np.ones(len(src) - tau)),
        None,
    )
    node.params.PhiID.mode.value = mode
    node.params.PhiID.tgt_index.value = tgt_index
    return node


def test_one_vs_others_rejects_out_of_bounds_tgt_index():
    node = _make_node(tgt_index=5)  # only 3 channels exist -> index 5 is out of range
    data = Data(DataType.ARRAY, np.random.rand(3, 128).astype(np.float32), {})

    with pytest.raises(ValueError, match="tgt_index"):
        node(matrix=data)


def test_one_vs_others_requires_at_least_two_channels():
    node = _make_node(tgt_index=0)  # index in range, but a single channel has no "others"
    data = Data(DataType.ARRAY, np.random.rand(1, 128).astype(np.float32), {})

    with pytest.raises(ValueError, match="at least 2 channels"):
        node(matrix=data)


def test_one_vs_others_valid_index_passes_bounds_check():
    # A valid in-bounds index with enough channels must NOT trip the bounds check; with
    # the stubbed calculator the full process completes and returns the three slots.
    node = _make_node(tgt_index=1)
    data = Data(DataType.ARRAY, np.random.rand(3, 128).astype(np.float32), {})

    out = node(matrix=data)

    assert set(out) == {"PhiID", "inf_dyn", "IIT"}
