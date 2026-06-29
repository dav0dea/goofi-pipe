"""Array/analysis nodes must propagate output metadata correctly AND never mutate the
producer's meta (fan-out aliasing, backlog #6 / §K). These nodes had TODO-marked gaps:

  - Transpose computed transposed channel names but returned an EMPTY meta dict (dropped),
    and mutated array.meta["channels"] in place (aliasing).
  - Reshape `del array.meta["channels"]` mutated the producer's meta in place.
  - PowerBand returned the FULL input meta, leaving the summed-out frequency axis as a
    stale channels entry on the reduced output.
"""
from copy import deepcopy

import numpy as np

from goofi.data import Data, DataType
from goofi.node import NodeEnv


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def test_transpose_propagates_transposed_channels_without_mutating_producer():
    from goofi.nodes.array.transpose import Transpose

    node = _standalone(Transpose)
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = Data(DataType.ARRAY, arr, {"channels": {"dim0": ["a", "b"], "dim1": ["x", "y", "z"]}})
    before = deepcopy(data.meta["channels"])

    out = node(array=data)
    value, meta = out["out"]

    assert value.shape == (3, 2)  # transposed
    # the output carries the TRANSPOSED channels (not an empty meta)
    assert meta["channels"]["dim0"] == ["x", "y", "z"]
    assert meta["channels"]["dim1"] == ["a", "b"]
    # the producer's meta is untouched (no fan-out aliasing)
    assert data.meta["channels"] == before


def test_reshape_does_not_mutate_producer_channels_meta():
    from goofi.nodes.array.reshape import Reshape

    node = _standalone(Reshape)
    node.params.reshape.shape.value = "6"
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = Data(DataType.ARRAY, arr, {"sfreq": 100.0, "channels": {"dim0": ["a", "b"], "dim1": ["x", "y", "z"]}})
    before = deepcopy(data.meta["channels"])

    out = node(array=data)
    value, meta = out["out"]

    assert value.shape == (6,)
    assert data.meta["channels"] == before  # producer not mutated (was a `del` on the input)
    assert "channels" not in meta  # reshape drops channels on its OWN copy
    assert meta.get("sfreq") == 100.0  # other meta survives


def test_powerband_drops_the_summed_out_frequency_axis_from_channels():
    from goofi.nodes.analysis.powerband import PowerBand

    node = _standalone(PowerBand)
    node.params.powerband.f_min.value = 1.0
    node.params.powerband.f_max.value = 3.0
    psd = np.ones((2, 4), dtype=np.float32)  # 2 channels x 4 freqs
    freqs = [0.5, 1.0, 2.0, 3.0]
    data = Data(DataType.ARRAY, psd, {"channels": {"dim0": ["c0", "c1"], "dim1": freqs}})
    before = deepcopy(data.meta["channels"])

    out = node(data=data)
    value, meta = out["power"]

    assert value.shape == (2,)  # one power value per channel (freqs summed out)
    assert meta["channels"]["dim0"] == ["c0", "c1"]  # channel axis preserved
    assert "dim1" not in meta["channels"]  # the freq axis is gone from the reduced output
    assert data.meta["channels"] == before  # producer not mutated
