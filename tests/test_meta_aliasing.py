"""Signal nodes must not mutate a producer's nested meta on fan-out (backlog #6, §K).

A node that shallow-copies `data.meta` then writes a NESTED key (e.g.
`meta['channels']['dimN'] = ...`) mutates the dict the producer still holds — so on a
fan-out every consumer/sibling sees the change. The fix is a deep copy before mutation.
"""
from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import NodeEnv


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def test_psd_does_not_mutate_producer_channels_meta():
    from goofi.nodes.signal.psd import PSD

    node = _standalone(PSD)
    node.params.psd.method.value = "fft"  # avoid the welch/scipy setup path

    sig = np.random.rand(128).astype(np.float32)
    data = Data(DataType.ARRAY, sig, {"sfreq": 128.0, "channels": {"dim0": list(range(128))}})
    before = deepcopy(data.meta["channels"])

    out = node(data=data)

    assert out["psd"][0].shape[0] > 0
    assert data.meta["channels"] == before, "PSD mutated the producer's channels meta (shallow-copy aliasing)"


def test_padding_does_not_mutate_producer_channels_meta():
    from goofi.nodes.array.padding import Padding

    node = _standalone(Padding)
    node.params.padding.pad_before.value = 2  # actually pad so the channels rewrite fires
    sig = np.random.rand(2, 64).astype(np.float32)
    data = Data(
        DataType.ARRAY,
        sig,
        {"channels": {"dim0": ["a", "b"], "dim1": list(range(64))}},
    )
    before = deepcopy(data.meta["channels"])

    node(array=data)

    assert data.meta["channels"] == before, "Padding mutated the producer's channels meta"
