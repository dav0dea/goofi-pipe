import time
from multiprocessing import Manager as MPManager
from multiprocessing import Process

import pytest

from goofi.connection import Connection
from goofi.message import Message, MessageType


@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_abstract_create(backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    with pytest.raises(TypeError):
        Connection._create()


@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_create(backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    conn1, conn2 = Connection.create()
    assert conn1 is not None and conn2 is not None, f"{backend.__name__}.create() returned None"


@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_super_init(backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    conn1, conn2 = Connection.create()
    assert hasattr(conn1, "_id"), f"{backend.__name__} does not call super().__init__()."
    assert hasattr(conn2, "_id"), f"{backend.__name__} does not call super().__init__()."


@pytest.mark.parametrize("obj", [1, "test", None, [1, 2, 3], {"a": 1}, Message(MessageType.PING, {})])
@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_send_recv(obj, backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    conn1, conn2 = Connection.create()
    conn1.send(obj)
    assert conn2.recv() == obj, "Connection.send() and Connection.recv() didn't work"


@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_close(backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    conn1, conn2 = Connection.create()

    # make sure the connection is not closed
    conn1.send(1)
    assert conn2.recv() == 1, "Connection.send() and Connection.recv() didn't work"

    conn1.close()

    # closing multiple times should not raise an error
    conn1.close()

    with pytest.raises(ConnectionError):
        conn1.send(1)
    with pytest.raises(ConnectionError):
        conn1.recv()

    conn2.close()
    conn2.close()


@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_try_send(backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    conn1, conn2 = Connection.create()

    assert conn2.try_send(1), "Connection.try_send() should return True on non-empty connection"
    assert conn1.recv() == 1, "Connection.send() and Connection.recv() didn't work"

    conn1.close()
    conn2.close()


@pytest.mark.parametrize("backend", Connection.get_ipc_backends().keys())
def test_multiproc(backend):
    try:
        mp_manager = MPManager()
        Connection.set_backend("mp", mp_manager)
    except AssertionError:
        # connection backend is already set
        pass

    def _send(conn):
        conn.send(1)
        # wait for the other process to receive the message before closing the connection
        time.sleep(0.1)

    conn1, conn2 = Connection.create()
    p = Process(target=_send, args=(conn1,))
    p.start()

    assert conn2.recv() == 1, "Connection.send() and Connection.recv() didn't work"
    p.join()
