import numpy as np

from goofi.audio.continuity import INDEX_META_KEY, is_discontinuous
from goofi.data import Data, DataType
from tests.utils import make_custom_node


def test_generator_gets_fresh_monotonic_index():
	# No input slot carries `index` -> each output emit gets a fresh counter.
	cls = make_custom_node(output_slots={"out": DataType.ARRAY})
	node = cls.create_standalone()
	assert [node._next_index("out") for _ in range(3)] == [0, 1, 2]


def test_two_outputs_have_independent_fresh_counters():
	cls = make_custom_node(output_slots={"a": DataType.ARRAY, "b": DataType.ARRAY})
	node = cls.create_standalone()
	assert node._next_index("a") == 0
	assert node._next_index("b") == 0
	assert node._next_index("a") == 1


def test_single_input_propagates_its_index():
	# Exactly one input slot holds Data with `index` -> propagate it, unchanged.
	cls = make_custom_node(
		input_slots={"signal": DataType.ARRAY},
		output_slots={"out": DataType.ARRAY},
	)
	node = cls.create_standalone()
	node.input_slots["signal"].data = Data(
		DataType.ARRAY, np.arange(4, dtype=np.float32), {INDEX_META_KEY: 42}
	)
	assert node._next_index("out") == 42
	# Propagation mirrors the input; it does not advance a local counter.
	assert node._next_index("out") == 42


def test_upstream_drop_makes_sink_visible_index_jump():
	cls = make_custom_node(
		input_slots={"signal": DataType.ARRAY},
		output_slots={"out": DataType.ARRAY},
	)
	node = cls.create_standalone()
	slot = node.input_slots["signal"]
	slot.data = Data(DataType.ARRAY, np.zeros(4, np.float32), {INDEX_META_KEY: 5})
	first = node._next_index("out")
	# 6 was lost on the wire upstream; the input's index jumps to 7.
	slot.data = Data(DataType.ARRAY, np.zeros(4, np.float32), {INDEX_META_KEY: 7})
	second = node._next_index("out")
	assert (first, second) == (5, 7)
	assert is_discontinuous(first, second)


def test_two_inputs_carrying_index_fall_back_to_fresh():
	# Ambiguous origin (not exactly one) -> fresh per-output counter.
	cls = make_custom_node(
		input_slots={"a": DataType.ARRAY, "b": DataType.ARRAY},
		output_slots={"out": DataType.ARRAY},
	)
	node = cls.create_standalone()
	node.input_slots["a"].data = Data(DataType.ARRAY, np.zeros(2, np.float32), {INDEX_META_KEY: 100})
	node.input_slots["b"].data = Data(DataType.ARRAY, np.zeros(2, np.float32), {INDEX_META_KEY: 200})
	assert node._propagated_index() is None
	assert node._next_index("out") == 0


def test_build_output_stamps_fresh_index_for_generator():
	cls = make_custom_node(output_slots={"out": DataType.ARRAY})
	node = cls.create_standalone()
	slot = node.output_slots["out"]
	d0 = node._build_output_data("out", slot, (np.zeros(2, np.float32), {}))
	d1 = node._build_output_data("out", slot, (np.zeros(2, np.float32), {}))
	assert d0.meta[INDEX_META_KEY] == 0
	assert d1.meta[INDEX_META_KEY] == 1
	assert d0.meta is not d1.meta  # each output slot builds its own Data


def test_build_output_fanout_shares_index_without_aliasing_input_meta():
	# Two outputs propagate the same single-input index; the built metas are
	# distinct dicts and the producer's input meta is left untouched.
	cls = make_custom_node(
		input_slots={"signal": DataType.ARRAY},
		output_slots={"a": DataType.ARRAY, "b": DataType.ARRAY},
	)
	node = cls.create_standalone()
	node.input_slots["signal"].data = Data(
		DataType.ARRAY, np.zeros(2, np.float32), {INDEX_META_KEY: 9}
	)
	a = node._build_output_data("a", node.output_slots["a"], (np.zeros(2, np.float32), {}))
	b = node._build_output_data("b", node.output_slots["b"], (np.zeros(2, np.float32), {}))
	assert a.meta[INDEX_META_KEY] == b.meta[INDEX_META_KEY] == 9
	assert a.meta is not b.meta
	assert node.input_slots["signal"].data.meta[INDEX_META_KEY] == 9
