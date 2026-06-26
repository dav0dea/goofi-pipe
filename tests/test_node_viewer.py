"""Phase 2 (node viewer-publish path) tests — Option C node-side reduction.

Starts with OutputSlot.viewer_count + lazy viewer_lock (pickle-safe gating state).
"""
import copy
import pickle
import threading

import numpy as np
import pytest

from goofi.data import Data, DataType
from goofi.node_helpers import OutputSlot
from goofi.codec import decode_data
from goofi.node_reduce import viewspec_from_dict


@pytest.fixture(autouse=True)
def _reset_node_viewer():
    """Stop + clear the reducer subsystem after every test (no thread leak)."""
    import goofi.node_viewer as nv

    yield
    nv._reset_for_tests()


class _Sink:
    """Collects delivered (encoded) frames and signals each arrival."""

    def __init__(self):
        self.frames = []
        self.event = threading.Event()
        self.gate = None  # optional threading.Event to block inside the sink

    def __call__(self, buf: bytes):
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        self.frames.append(buf)
        self.event.set()

    def wait(self, n=1, timeout=5.0):
        # wait until at least n frames have arrived
        import time

        deadline = time.monotonic() + timeout
        while len(self.frames) < n and time.monotonic() < deadline:
            self.event.wait(timeout=0.05)
            self.event.clear()
        return len(self.frames)


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


# ---- node_viewer reducer subsystem -------------------------------------------

def _big_line(n=10000):
    return Data(DataType.ARRAY, np.linspace(-1, 1, n).astype(np.float32),
                {"channels": {"dim0": list(range(n))}})


def test_snapshot_for_offer_does_not_alias_input():
    import goofi.node_viewer as nv

    arr = np.ones(8, dtype=np.float32)
    data = Data(DataType.ARRAY, arr, {})
    snap = nv._snapshot_for_offer(data)
    arr[:] = 999.0  # mutate the node's array in place AFTER the snapshot
    assert np.all(snap.data == 1.0)  # snapshot is an independent copy
    assert snap.meta["channels"] is not data.meta["channels"]


def test_offer_reduces_to_spec_and_delivers_encoded_frame():
    import goofi.node_viewer as nv

    sink = _Sink()
    nv.register_viewer("n0", "out", sink)
    nv.set_viewspec("n0", "out", viewspec_from_dict(
        {"axes": [{"axis": -1, "max": 1000, "method": "envelope"}]}))
    nv.offer("n0", "out", _big_line(10000))
    assert sink.wait(1) == 1
    red = decode_data(sink.frames[-1])
    assert red.data.shape == (2000,)  # 2*W envelope, reduced inside the reducer thread


def test_latest_wins_coalesces_while_reducer_busy():
    import goofi.node_viewer as nv

    sink = _Sink()
    sink.gate = threading.Event()  # block the reducer inside the first sink call
    nv.register_viewer("n1", "out", sink)
    nv.set_viewspec("n1", "out", viewspec_from_dict(
        {"axes": [{"axis": -1, "max": 1000, "method": "envelope"}]}))

    # Constant-fill frames so the marker survives envelope (min==max==v).
    def frame(v):
        return Data(DataType.ARRAY, np.full(10000, v, dtype=np.float32), {})

    # Frame 1: reducer picks it up and blocks in the sink.
    nv.offer("n1", "out", frame(-1.0))
    # Wait until the reducer is parked inside the sink (frame in flight).
    import time
    time.sleep(0.2)

    # Frames 2..4 queue while the reducer is busy -> latest-wins coalesces them.
    for v in (0.1, 0.2, 0.5):
        nv.offer("n1", "out", frame(v))

    sink.gate.set()  # release the reducer
    assert sink.wait(2) == 2  # exactly two frames: the first + the coalesced latest
    last = decode_data(sink.frames[-1])
    assert abs(float(last.data[0]) - 0.5) < 1e-3  # latest (0.5) won, not 0.1/0.2


def test_inplace_mutation_after_offer_cannot_affect_delivered_frame():
    import goofi.node_viewer as nv

    sink = _Sink()
    nv.register_viewer("n2", "out", sink)
    nv.set_viewspec("n2", "out", viewspec_from_dict(
        {"axes": [{"axis": -1, "max": 100, "method": "envelope"}]}))
    d = Data(DataType.ARRAY, np.zeros(1000, dtype=np.float32), {})
    nv.offer("n2", "out", d)
    d.data[:] = 999.0  # LatentRotator-shaped in-place mutation after return
    assert sink.wait(1) == 1
    red = decode_data(sink.frames[-1])
    assert float(np.max(np.abs(red.data))) == 0.0  # delivered the snapshot (zeros)


def test_reducer_survives_a_failing_sink():
    import goofi.node_viewer as nv

    calls = {"n": 0}

    def flaky(buf):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")  # first delivery explodes

    nv.register_viewer("n3", "out", flaky)
    nv.set_viewspec("n3", "out", viewspec_from_dict(
        {"axes": [{"axis": -1, "max": 100, "method": "envelope"}]}))
    nv.offer("n3", "out", _big_line(5000))
    import time
    time.sleep(0.3)
    nv.offer("n3", "out", _big_line(5000))  # reducer must still be alive
    deadline = time.monotonic() + 5.0
    while calls["n"] < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert calls["n"] == 2


def test_evict_clears_state_and_drops_pending():
    import goofi.node_viewer as nv

    sink = _Sink()
    nv.register_viewer("n4", "out", sink)
    nv.set_viewspec("n4", "out", viewspec_from_dict(
        {"axes": [{"axis": -1, "max": 100, "method": "envelope"}]}))
    nv.evict("n4", "out")
    assert ("n4", "out") not in nv._sinks
    assert ("n4", "out") not in nv._specs
    nv.offer("n4", "out", _big_line(2000))  # no sink -> dropped, no crash
    assert ("n4", "out") not in nv._pending


def test_reset_for_tests_stops_reducer_thread():
    import goofi.node_viewer as nv

    nv.register_viewer("n5", "out", _Sink())
    nv.offer("n5", "out", _big_line(2000))
    nv._reset_for_tests()
    assert not any(t.name == "goofi-data-reducer" and t.is_alive()
                   for t in threading.enumerate())


# ---- node.py integration: viewer-publish path (real spawned node) ------------

def test_viewer_path_delivers_reduced_frames_end_to_end():
    """A browser-only slot (subscriber_count==0) produces reduced frames on its
    '.view' iceoryx2 service: validates the OR-gate, offer(), the reducer thread,
    and the cross-thread+cross-process iceoryx2 publish in one shot."""
    import time as _t

    from goofi.codec import decode_data as _decode
    from goofi.message import Message, MessageType
    from goofi.transport import data_service_name, open_subscriber

    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        # Manager-side subscriber on the node's reduced viewer service.
        svc = data_service_name(ref.node_id, "out") + ".view"
        sub, _listener = open_subscriber(svc, in_process=False, latest_wins=True)

        # Drive the new ctrl messages directly (NodeRef wrappers land in Phase 3).
        ref._send(Message(MessageType.SET_VIEWSPEC, {
            "slot_name_out": "out",
            "spec": {"axes": [{"axis": -1, "max": 200, "method": "envelope"}]},
        }))
        ref._send(Message(MessageType.REGISTER_VIEWER, {"slot_name_out": "out"}))

        buf = None
        deadline = _t.time() + 6.0
        while buf is None and _t.time() < deadline:
            buf = sub.take_latest()
            if buf is None:
                _t.sleep(0.02)
        assert buf is not None, "no reduced frame arrived on the .view service"
        red = _decode(buf)
        assert red.dtype == DataType.ARRAY  # reduced GOOF frame decodes cleanly

        # Tear the viewer down — must not raise.
        ref._send(Message(MessageType.UNREGISTER_VIEWER, {"slot_name_out": "out"}))
        _t.sleep(0.1)
    finally:
        mgr.terminate(notify_gui=False)


# ---- Phase 3a: NodeRef viewer API (manager-side) -----------------------------

def test_noderef_view_handler_delivers_reduced_frames():
    """The NodeRef viewer API end-to-end: set_viewspec + a view-plane data handler
    (REGISTER_VIEWER + subscribe to .view + raw pump) delivers reduced frames."""
    import time as _t

    from goofi.codec import decode_data as _decode
    from .test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        osc = mgr.add_node("Oscillator", "inputs")
        ref = mgr.nodes[osc]
        ref.wait_for_state(timeout=2.0)

        ref.set_viewspec("out", {
            "axes": [{"axis": -1, "max": 200, "method": "envelope"}], "version": 1})
        frames = []
        ref.set_data_handler("out", lambda _nr, _s, buf: frames.append(buf),
                             raw=True, view=True)

        deadline = _t.time() + 6.0
        while not frames and _t.time() < deadline:
            _t.sleep(0.02)
        ref.set_data_handler("out", None, view=True)

        assert frames, "view handler received no reduced frames"
        red = _decode(frames[-1])
        assert red.dtype == DataType.ARRAY
    finally:
        mgr.terminate(notify_gui=False)
