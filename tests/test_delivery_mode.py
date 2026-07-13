"""Delivery-mode (queue vs latest) wiring on the node side."""
import numpy as np

from goofi.data import Data, DataType
from goofi.node_helpers import InputSlot
from goofi.transport import ThreadListener, set_instance_id
from tests.utils import make_custom_node


def _fresh_iid():
    import uuid

    set_instance_id(f"dm-{uuid.uuid4().hex[:8]}")


def test_subscribe_input_threads_queue_and_buffer_size(monkeypatch):
    _fresh_iid()
    from goofi import node as node_mod

    captured = {}

    class _FakeSub:
        def close(self):
            pass

    def fake_open_subscriber(name, *, in_process, safe_overflow=True, buffer_cap=None, buffer_size=None):
        captured.update(name=name, buffer_cap=buffer_cap, buffer_size=buffer_size)
        return _FakeSub(), ThreadListener(name + ".evt")

    monkeypatch.setattr(node_mod, "open_subscriber", fake_open_subscriber)

    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY})
    n = Cls.create_standalone()

    n._subscribe_input(slot_name_in="data", service_name="svc-q", in_process=False, queue=True, buffer_cap=8)
    assert n.input_slots["data"].queue is True
    assert captured["buffer_cap"] == 8 and captured["buffer_size"] == 8

    n._subscribe_input(slot_name_in="data", service_name="svc-l", in_process=False, queue=False, buffer_cap=None)
    assert n.input_slots["data"].queue is False
    assert captured["buffer_size"] == 2


def test_ensure_output_endpoints_threads_buffer_cap(monkeypatch):
    _fresh_iid()
    from goofi import node as node_mod

    captured = {}

    class _FakePub:
        def close(self):
            pass

    class _FakeNotif:
        def close(self):
            pass

    def fake_open_publisher(name, *, in_process, safe_overflow=True, buffer_cap=None):
        captured.update(name=name, buffer_cap=buffer_cap)
        return _FakePub(), _FakeNotif()

    monkeypatch.setattr(node_mod, "open_publisher", fake_open_publisher)

    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY})
    n = Cls.create_standalone()

    n._ensure_output_endpoints("out", want_ipc=True, buffer_cap=8)
    assert captured["buffer_cap"] == 8
    assert n.output_slots["out"].ipc_buffer_cap == 8


def test_dispatch_burst_runs_process_per_queued_frame_holding_control():
    _fresh_iid()
    seen = []

    def cb(**kw):
        seen.append((float(kw["data"].data[0]), float(kw["ctrl"].data[0])))
        return {"out": (kw["data"].data, {})}

    Cls = make_custom_node(
        input_slots={"data": InputSlot(DataType.ARRAY, queue=True), "ctrl": DataType.ARRAY},
        output_slots={"out": DataType.ARRAY},
        process_callback=cb,
    )
    n = Cls.create_standalone()
    n.input_slots["ctrl"].data = Data(DataType.ARRAY, np.array([9.0]), {})

    frames = [Data(DataType.ARRAY, np.array([float(i)]), {}) for i in range(3)]
    n._dispatch_burst({"data": frames})

    assert len(seen) == 3, f"expected 3 process() calls, got {len(seen)}"
    assert [d for d, _ in seen] == [0.0, 1.0, 2.0], "queue slot did not advance frame-by-frame"
    assert all(c == 9.0 for _, c in seen), "control input was not held across the burst"


def test_dispatch_burst_latest_mode_calls_process_once():
    _fresh_iid()
    calls = []

    def cb(**kw):
        calls.append(1)
        return None

    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY}, process_callback=cb)
    n = Cls.create_standalone()
    n.input_slots["data"].data = Data(DataType.ARRAY, np.array([1.0]), {})
    n._dispatch_burst({})  # no fired queue slot → burst == 1
    assert len(calls) == 1


def test_drain_queue_decodes_all_pending_in_order():
    _fresh_iid()
    from goofi.codec import encode_data

    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY})
    n = Cls.create_standalone()

    bufs = [encode_data(Data(DataType.ARRAY, np.array([float(i)]), {})) for i in range(4)]

    class _FakeSub:
        def __init__(self, b):
            self._b = list(b)

        def take_next(self):
            return self._b.pop(0) if self._b else None

    frames = n._drain_queue(_FakeSub(bufs))
    assert [float(f.data[0]) for f in frames] == [0.0, 1.0, 2.0, 3.0]


def _wait_until(pred, timeout=10.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_mixed_mode_fanout_shares_one_service_cap():
    """A source slot fanning out to a LATEST consumer AND a QUEUE consumer must
    provision ONE shared iceoryx2 service cap (the aggregate max), regardless of
    wiring order. subscriber_max_buffer_size is a static per-service minimum, so a
    per-consumer cap makes the deeper (queue) request fail on the service the
    latest tap already created at the default cap — the queue consumer then gets
    no data while the source silently publishes to an empty publisher list."""
    import time

    from goofi.manager import QUEUE_DEPTH
    from tests.test_manager import _bare_manager

    mgr = _bare_manager()  # real multiprocessing: exercises the iceoryx2 service
    try:
        src = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        latest = mgr.add_node("Buffer", "signal")   # `val` input: latest-wins (default)
        queued = mgr.add_node("Gain", "signal")     # `signal` input: queue=True
        assert _wait_until(lambda: all(mgr.nodes[u].stage == "ready" for u in (src, latest, queued)))

        # Wire the LATEST consumer FIRST (creates the service at the default cap),
        # then the QUEUE consumer (needs the deeper cap on that same service).
        mgr.add_link(src, latest, "out", "val")
        mgr.add_link(src, queued, "out", "signal")

        # The source must keep a live publisher for BOTH consumers...
        assert _wait_until(
            lambda: (mgr.nodes[src].serialized_state or {}).get("output_subscribers", {}).get("out", 0) == 2
        )
        # ...and no consumer may carry a subscribe error from an incompatible cap.
        time.sleep(0.5)
        assert not mgr.nodes[queued].last_error, f"queue consumer errored: {mgr.nodes[queued].last_error}"
        assert not mgr.nodes[latest].last_error, f"latest consumer errored: {mgr.nodes[latest].last_error}"
        assert not mgr.nodes[src].last_error, f"source errored: {mgr.nodes[src].last_error}"
    finally:
        mgr.terminate(notify_gui=False)


def test_live_mode_flip_reallocates_whole_source_slot():
    """Flipping one consumer latest->queue (set_node_inputs) must reallocate the
    WHOLE source slot: a sibling latest consumer on the same slot pins the shared
    service at the shallow cap, so reallocating only the flipped consumer's link
    would fail the deeper reopen. Both consumers must survive the flip."""
    import time

    from tests.test_manager import _bare_manager

    mgr = _bare_manager()
    try:
        src = mgr.add_node("Oscillator", "inputs", params={"common": {"autotrigger": True}})
        a = mgr.add_node("Buffer", "signal")   # latest
        b = mgr.add_node("Buffer", "signal")   # latest (sibling that pins the service)
        assert _wait_until(lambda: all(mgr.nodes[u].stage == "ready" for u in (src, a, b)))
        mgr.add_link(src, a, "out", "val")
        mgr.add_link(src, b, "out", "val")
        assert _wait_until(
            lambda: (mgr.nodes[src].serialized_state or {}).get("output_subscribers", {}).get("out", 0) == 2
        )
        # Flip consumer `a` to queue mode while `b` stays latest.
        mgr.set_node_inputs(a, {"val": {"queue": True}})
        time.sleep(0.5)
        for uid in (src, a, b):
            assert not mgr.nodes[uid].last_error, f"{uid} errored after flip: {mgr.nodes[uid].last_error}"
        assert (mgr.nodes[src].serialized_state or {}).get("output_subscribers", {}).get("out", 0) == 2
    finally:
        mgr.terminate(notify_gui=False)
