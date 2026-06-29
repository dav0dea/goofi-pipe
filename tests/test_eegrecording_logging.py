"""EEGRecording must warn via the logging module, not print().

The codebase convention (see src/goofi/manager.py) is a module-level
`logger = logging.getLogger(__name__)` with `logger.warning(...)`. When both
`use_example_data` and `file_path` are set, `setup()` should emit a WARNING log
record (capturable by `caplog`) rather than writing to stdout via print().
"""
import logging

import pytest

from goofi.node import NodeEnv
from goofi.nodes.inputs.eegrecording import EEGRecording

LOGGER_NAME = "goofi.nodes.inputs.eegrecording"


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def test_setup_logs_warning_when_both_example_data_and_file_path_set(caplog):
    node = _standalone(EEGRecording)
    node.params.recording.use_example_data.value = True
    node.params.recording.file_path.value = "/nonexistent/does_not_exist.fif"

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        # After emitting the warning, setup() tries to load the (nonexistent)
        # file and raises. We only care that the warning came through logging,
        # not print(), so swallow the load failure.
        with pytest.raises(Exception):
            node.setup()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "Both 'use_example_data' and 'file_path' are set" in r.getMessage() for r in warnings
    ), "expected a WARNING log record about both flags being set (was it print()ed instead?)"
