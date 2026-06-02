"""Tests for the iceoryx2 + thread transport layer.

Each test sets a fresh instance id so concurrent test runs (and stale
shared-memory entries from previous runs) don't collide.
"""
from __future__ import annotations

import threading
import time
import uuid

import pytest

from goofi.codec import decode_data, decode_message, encode_data, encode_message
from goofi.data import Data, DataType
from goofi.message import Message, MessageType
from goofi.transport import (
    IpcListener,
    ThreadListener,
    ThreadNotifier,
    ThreadPublisher,
    ThreadSubscriber,
    WaitSet,
    data_service_name,
    open_publisher,
    open_subscriber,
    set_instance_id,
)

_EVT = ".evt"

import numpy as np


def _fresh_iid():
    set_instance_id(f"t-{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Codec roundtrip — exercises the (de)serialization path the transport uses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [DataType.ARRAY, DataType.STRING, DataType.TABLE])
def test_codec_data_roundtrip(dtype):
    if dtype == DataType.ARRAY:
        d = Data(dtype, np.arange(12, dtype=np.float32).reshape(3, 4), {"sfreq": 256})
    elif dtype == DataType.STRING:
        d = Data(dtype, "hello", {"lang": "en"})
    else:
        d = Data(
            dtype,
            {
                "a": Data(DataType.ARRAY, np.zeros(3), {}),
                "b": Data(DataType.STRING, "foo", {}),
            },
            {"note": "test"},
        )

    buf = encode_data(d)
    d2 = decode_data(buf)
    assert d2.dtype == d.dtype
    if dtype == DataType.ARRAY:
        assert np.array_equal(d.data, d2.data)
        assert d.data.dtype == d2.data.dtype
    elif dtype == DataType.STRING:
        assert d.data == d2.data
    else:
        assert set(d.data) == set(d2.data)
        assert isinstance(d2.data["a"], Data)


def test_codec_meta_exceeds_16bit():
    """meta_len is a u32 field: a node can carry a large coordinate array in
    meta. PSD/FFT store the full frequency axis under channels["dim<axis>"],
    which exceeds 64 KB at fine resolution — this used to overflow the old u16
    meta_len with `struct.error: 'H' format requires 0 <= number <= 65535`."""
    freq = np.linspace(0.0, 22050.0, 55123)  # ~496 KB once msgpack-encoded
    d = Data(DataType.ARRAY, np.random.rand(55123), {"channels": {"dim0": freq.tolist()}})
    d2 = decode_data(encode_data(d))
    assert np.array_equal(d.data, d2.data)
    assert d2.meta["channels"]["dim0"] == pytest.approx(freq.tolist())


@pytest.mark.parametrize(
    "msg",
    [
        Message(MessageType.PARAMETER_UPDATE, {"group": "common", "param_name": "x", "param_value": 1.5}),
        Message(MessageType.SUBSCRIBE_INPUT, {"slot_name_in": "in", "service_name": "s", "in_process": True}),
        Message(MessageType.REGISTER_SUBSCRIBER, {"slot_name_out": "out"}),
        Message(MessageType.STATE_UPDATE, {"_type": "X", "category": "y", "params": {}, "output_subscribers": {}}),
        Message(MessageType.TERMINATE, {}),
        Message(MessageType.SHUTDOWN, {}),
    ],
)
def test_codec_message_roundtrip(msg):
    m2 = decode_message(encode_message(msg))
    assert m2.type == msg.type
    assert m2.content == msg.content


# ---------------------------------------------------------------------------
# Ipc transport
# ---------------------------------------------------------------------------


def test_ipc_pubsub_latest_wins():
    _fresh_iid()
    name = data_service_name("nodeA", "out")
    pub, notif = open_publisher(name, in_process=False, latest_wins=True)
    sub, listener = open_subscriber(name, in_process=False, latest_wins=True)
    for i in range(10):
        pub.send(f"msg{i}".encode())
        notif.notify()
    time.sleep(0.05)
    assert sub.take_latest() == b"msg9"


def test_ipc_atomic_instance_array():
    _fresh_iid()
    name = data_service_name("nodeAtomic", "out")
    pub, _notif = open_publisher(name, in_process=False, latest_wins=True)
    sub, _list = open_subscriber(name, in_process=False, latest_wins=True)

    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    d = Data(DataType.ARRAY, arr, {})
    pub.send(encode_data(d))
    time.sleep(0.05)
    # Mutate sender's array after send
    arr[0] = 99
    got = sub.take_latest()
    assert got is not None
    decoded = decode_data(got)
    assert decoded.data[0] == 1, "subscriber saw producer's later mutation"


def test_ipc_waitset_wake():
    _fresh_iid()
    name = data_service_name("nodeWS", "out")
    pub, notif = open_publisher(name, in_process=False, latest_wins=True)
    _sub, listener = open_subscriber(name, in_process=False, latest_wins=True)
    ws = WaitSet()
    ws.attach(listener)

    def fire():
        time.sleep(0.05)
        pub.send(b"x")
        notif.notify()

    threading.Thread(target=fire, daemon=True).start()
    t0 = time.time()
    fired = ws.wait(1.0)
    elapsed = time.time() - t0
    assert fired, "WaitSet did not wake"
    assert elapsed < 0.5, f"WaitSet wake took too long: {elapsed:.3f}s"


def test_ipc_waitset_timeout():
    _fresh_iid()
    _sub, listener = open_subscriber(data_service_name("idle", "out"), in_process=False, latest_wins=True)
    ws = WaitSet()
    ws.attach(listener)
    t0 = time.time()
    fired = ws.wait(0.2)
    elapsed = time.time() - t0
    assert not fired
    assert 0.15 < elapsed < 0.4, f"unexpected elapsed: {elapsed:.3f}"


def test_ctrl_channel_order_under_burst():
    _fresh_iid()
    name = f"goofi-test-ctrl-{uuid.uuid4().hex[:8]}"
    pub, notif = open_publisher(name, in_process=False, latest_wins=False)
    sub, listener = open_subscriber(name, in_process=False, latest_wins=False)

    # Burst N messages; expect them all in order.
    N = 20
    for i in range(N):
        pub.send(str(i).encode())
        notif.notify()
    time.sleep(0.1)

    received = []
    while True:
        buf = sub.take_next()
        if buf is None:
            break
        received.append(buf.decode())
    assert received == [str(i) for i in range(N)], f"got {received}"


# ---------------------------------------------------------------------------
# Thread transport
# ---------------------------------------------------------------------------


def test_thread_pubsub_latest_wins():
    name = f"goofi-test-thread-{uuid.uuid4().hex[:8]}"
    pub = ThreadPublisher(name, latest_wins=True)
    sub = ThreadSubscriber(name, latest_wins=True)
    for i in range(5):
        pub.send(f"t{i}".encode())
    assert sub.take_latest() == b"t4"


def test_thread_atomic_instance_bytearray():
    name = f"goofi-test-thread-atom-{uuid.uuid4().hex[:8]}"
    pub = ThreadPublisher(name, latest_wins=True)
    sub = ThreadSubscriber(name, latest_wins=True)
    buf = bytearray(b"abc")
    pub.send(buf)
    buf[0] = ord("Z")
    assert sub.take_latest() == b"abc", "thread transport leaked sender's mutable buffer"


def test_thread_listener_wake():
    name = f"goofi-test-thread-wake-{uuid.uuid4().hex[:8]}"
    listener = ThreadListener((name + _EVT))
    notif = ThreadNotifier((name + _EVT))
    ws = WaitSet()
    ws.attach(listener)
    # Drain any pre-existing state.
    listener._consume()

    def fire():
        time.sleep(0.03)
        notif.notify()

    threading.Thread(target=fire, daemon=True).start()
    t0 = time.time()
    fired = ws.wait(1.0)
    elapsed = time.time() - t0
    assert fired
    assert elapsed < 0.3, f"thread wake too slow: {elapsed:.3f}s"


def test_thread_ctrl_order():
    """Reliable thread ctrl: messages arrive in order, no drops."""
    name = f"goofi-test-thread-ctrl-{uuid.uuid4().hex[:8]}"
    pub, _ = open_publisher(name, in_process=True, latest_wins=False)
    sub, _ = open_subscriber(name, in_process=True, latest_wins=False)
    for i in range(8):
        pub.send(str(i).encode())
    received = []
    while True:
        b = sub.take_next()
        if b is None:
            break
        received.append(b.decode())
    assert received == [str(i) for i in range(8)]
