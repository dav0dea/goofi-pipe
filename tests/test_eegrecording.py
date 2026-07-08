"""EEGRecording emits recording samples directly on its own tick (no LSL)."""
import numpy as np
import pytest

from goofi.data import DataType
from goofi.nodes.inputs.eegrecording import EEGRecording

# 2 channels x 5 samples, distinct values so wrapping is observable.
DATA = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 14.0]])


def test_read_chunk_loops_and_wraps_at_boundary():
    chunk, new_idx = EEGRecording._read_chunk(DATA, 3, 4, loop=True)
    # indices [3, 4, 0, 1]
    assert np.array_equal(chunk, DATA[:, [3, 4, 0, 1]])
    assert new_idx == (3 + 4) % 5  # 2


def test_read_chunk_handles_n_larger_than_recording():
    chunk, new_idx = EEGRecording._read_chunk(DATA, 0, 12, loop=True)
    assert chunk.shape == (2, 12)
    assert np.array_equal(chunk, DATA[:, np.arange(0, 12) % 5])
    assert new_idx == 12 % 5  # 2


def test_read_chunk_no_loop_clamps_to_end():
    chunk, new_idx = EEGRecording._read_chunk(DATA, 3, 4, loop=False)
    assert np.array_equal(chunk, DATA[:, 3:5])  # only 2 samples remain
    assert new_idx == 5


def test_read_chunk_no_loop_past_end_returns_none():
    chunk, new_idx = EEGRecording._read_chunk(DATA, 5, 3, loop=False)
    assert chunk is None
    assert new_idx == 5


import goofi.nodes.inputs.eegrecording as eeg_mod


def _make_csv(tmp_path):
    # read_csv(index_col=0) -> 2 float channels (c0, c1), 4 samples.
    p = tmp_path / "rec.csv"
    p.write_text(",c0,c1\n0,0.1,1.1\n1,0.2,1.2\n2,0.3,1.3\n3,0.4,1.4\n")
    return str(p)


def _setup_from_csv(tmp_path, sfreq=100.0):
    node = EEGRecording.create_standalone()
    node.params.recording.use_example_data.value = False
    node.params.recording.file_path.value = _make_csv(tmp_path)
    node.params.recording.file_sfreq.value = sfreq
    node.setup()
    return node


def test_config_drops_lsl_params_and_adds_output():
    p = EEGRecording.config_params()
    assert "source_name" not in p["recording"]
    assert "stream_name" not in p["recording"]
    assert "loop" in p["recording"] and "reset" in p["recording"]
    assert p["common"]["autotrigger"] is True
    assert EEGRecording.config_output_slots()["out"] == DataType.ARRAY


def test_setup_loads_csv_into_data_sfreq_channels(tmp_path):
    node = _setup_from_csv(tmp_path, sfreq=100.0)
    assert node.data.shape == (2, 4)
    assert node.sfreq == 100.0
    assert node.ch_names == ["c0", "c1"]
    assert node.idx == 0


def test_process_emits_tick_paced_chunk_and_advances(tmp_path, monkeypatch):
    node = _setup_from_csv(tmp_path, sfreq=100.0)
    node.last_trigger = 100.0
    monkeypatch.setattr(eeg_mod.time, "time", lambda: 100.1)  # dt = 0.1 -> n = 10
    out = node.process()
    chunk, meta = out["out"]
    assert chunk.shape == (2, 10)  # n=10 > total=4, wraps (loop defaults True)
    assert meta["sfreq"] == 100.0
    assert meta["channels"]["dim0"] == ["c0", "c1"]
    assert node.idx == 10 % 4  # 2


def test_reset_zeroes_the_cursor(tmp_path):
    node = _setup_from_csv(tmp_path)
    node.idx = 3
    node.recording_reset_changed(True)
    assert node.idx == 0


def test_no_loop_past_end_emits_nothing(tmp_path, monkeypatch):
    node = _setup_from_csv(tmp_path, sfreq=100.0)
    node.params.recording.loop.value = False
    node.idx = 4  # at end
    node.last_trigger = 100.0
    monkeypatch.setattr(eeg_mod.time, "time", lambda: 100.1)  # n = 10 > 0
    assert node.process() is None
