import numpy as np

from goofi.audio.continuity import INDEX_META_KEY, crossfade, is_discontinuous


def test_index_meta_key_value():
    assert INDEX_META_KEY == "index"


def test_is_discontinuous_none_cases_are_false():
    assert is_discontinuous(None, 5) is False
    assert is_discontinuous(5, None) is False
    assert is_discontinuous(None, None) is False


def test_is_discontinuous_contiguous_vs_jump():
    assert is_discontinuous(4, 5) is False
    assert is_discontinuous(4, 7) is True
    assert is_discontinuous(4, 4) is True
    assert is_discontinuous(4, 2) is True


def test_crossfade_preserves_length_and_is_new_array():
    prev = np.ones((8, 2), dtype=np.float32)
    block = np.zeros((10, 2), dtype=np.float32)
    out = crossfade(prev, block, 4)
    assert out.shape == block.shape
    assert out is not block
    assert np.array_equal(block, np.zeros((10, 2), dtype=np.float32))  # input untouched


def test_crossfade_replaces_window_not_prepend():
    prev = np.ones((4, 1), dtype=np.float32)      # held value 1.0
    block = np.zeros((10, 1), dtype=np.float32)   # incoming 0.0
    out = crossfade(prev, block, 4)
    win = out[:4, 0]
    # equal-power fade from prev(1.0) into block(0.0): monotone decreasing, no inflation
    assert np.all(np.diff(win) < 0)
    assert win[0] > 0.8
    assert win[-1] < 0.5
    assert np.all(out[4:] == 0.0)  # untouched tail, same length (replace, not prepend)


def test_crossfade_zero_or_negative_window_is_passthrough_copy():
    prev = np.ones((4, 1), dtype=np.float32)
    block = np.arange(6, dtype=np.float32)[:, None]
    out = crossfade(prev, block, 0)
    assert out is not block
    assert np.array_equal(out, block)


def test_crossfade_window_larger_than_block_covers_whole_block():
    prev = np.ones((4, 1), dtype=np.float32)
    block = np.zeros((3, 1), dtype=np.float32)
    out = crossfade(prev, block, 999)
    assert out.shape == (3, 1)
    assert np.all(np.diff(out[:, 0]) < 0)
