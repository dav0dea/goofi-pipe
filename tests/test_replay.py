"""Replay must preserve column dtypes — integer/timestamp columns must not be
coerced to float. float32 cannot represent epoch-ms timestamps or large counters
(anything > 2^24), so blanket-coercing CSV numerics to float silently corrupts
them. Replay should let numpy infer each column's natural dtype; only genuine
floats are narrowed to float32 at the Data boundary (the single chokepoint)."""
import numpy as np

from goofi.node import NodeEnv
from goofi.nodes.inputs.replay import Replay, convert_to_numpy


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def _play(csv):
    node = _standalone(Replay)
    node.params["Read"]["filename"].value = str(csv)
    node.params["Read"]["play"].value = True
    node.setup()
    return node.process()["table_output"][0]


def test_convert_to_numpy_preserves_integer_lists():
    arr = convert_to_numpy("[1, 2, 3]")
    assert np.issubdtype(arr.dtype, np.integer)
    assert arr.tolist() == [1, 2, 3]


def test_convert_to_numpy_keeps_float_lists_float():
    arr = convert_to_numpy("[1.5, 2.5]")
    assert np.issubdtype(arr.dtype, np.floating)
    assert arr.tolist() == [1.5, 2.5]


def test_replay_preserves_large_integer_timestamp_column(tmp_path):
    ts = 1_750_000_000_123  # epoch-ms; unrepresentable in float32 (~56s of error)
    csv = tmp_path / "ts.csv"
    csv.write_text("timestamp,signal\n%d,0.5\n" % ts)
    out = _play(csv)
    assert int(out["timestamp"].data[0]) == ts
    assert np.issubdtype(out["timestamp"].data.dtype, np.integer)


def test_replay_preserves_large_integer_list_column(tmp_path):
    csv = tmp_path / "vec.csv"
    csv.write_text('idx\n"[1000000001, 1000000002, 1000000003]"\n')
    out = _play(csv)
    np.testing.assert_array_equal(out["idx"].data, [1000000001, 1000000002, 1000000003])
    assert np.issubdtype(out["idx"].data.dtype, np.integer)


def test_replay_narrows_float_columns_to_float32(tmp_path):
    csv = tmp_path / "f.csv"
    csv.write_text("signal\n0.5\n")
    out = _play(csv)
    assert out["signal"].data.dtype == np.float32
    assert float(out["signal"].data[0]) == 0.5
