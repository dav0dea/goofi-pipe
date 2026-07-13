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
