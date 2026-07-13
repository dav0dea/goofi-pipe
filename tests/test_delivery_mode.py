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

    def fake_open_publisher(name, *, in_process, safe_overflow=True, buffer_cap=None, max_subscribers=16):
        captured.update(name=name, buffer_cap=buffer_cap, max_subscribers=max_subscribers)
        return _FakePub(), _FakeNotif()

    monkeypatch.setattr(node_mod, "open_publisher", fake_open_publisher)

    Cls = make_custom_node(input_slots={"data": DataType.ARRAY}, output_slots={"out": DataType.ARRAY})
    n = Cls.create_standalone()

    n._ensure_output_endpoints("out", want_ipc=True, buffer_cap=8, max_subscribers=4)
    assert captured["buffer_cap"] == 8 and captured["max_subscribers"] == 4
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
