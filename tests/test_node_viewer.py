"""Phase 2 (node viewer-publish path) tests — Option C node-side reduction.

Starts with OutputSlot.viewer_count + lazy viewer_lock (pickle-safe gating state).
"""
import copy
import pickle
import threading

from goofi.data import DataType
from goofi.node_helpers import OutputSlot


def test_output_slot_viewer_count_default_zero():
    assert OutputSlot(DataType.ARRAY).viewer_count == 0


def test_output_slot_pickles_before_lock_creation():
    # OutputSlot is pickled (cls._configure -> MP spawn) BEFORE any viewer wiring,
    # so the lazy viewer_lock is never present at pickle time -> still picklable.
    slot = OutputSlot(DataType.ARRAY)
    slot.viewer_count = 2
    restored = pickle.loads(pickle.dumps(slot))
    assert restored.viewer_count == 2
    assert copy.deepcopy(slot).viewer_count == 2


def test_output_slot_viewer_lock_is_stable_and_a_lock():
    slot = OutputSlot(DataType.ARRAY)
    lk1 = slot.viewer_lock
    lk2 = slot.viewer_lock
    assert lk1 is lk2  # same lock object across calls
    # acquirable/releasable like a real lock
    assert lk1.acquire(blocking=False)
    lk1.release()


def test_output_slot_viewer_count_does_not_affect_equality():
    a = OutputSlot(DataType.ARRAY)
    b = OutputSlot(DataType.ARRAY)
    b.viewer_count = 5
    assert a == b  # viewer_count is compare=False bookkeeping
