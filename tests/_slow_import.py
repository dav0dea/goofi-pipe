"""Fixture module for spawn-lifecycle tests: importing it takes long enough to
hold the spawned node in its 'creating' stage while the test acts on it."""

import time

time.sleep(3.0)

from goofi.nodes.inputs.constantarray import ConstantArray as SlowNode  # noqa: E402,F401
