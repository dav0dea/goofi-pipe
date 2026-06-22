"""I/O node terminate() hygiene + delay interruptibility (backlog #7, §L).

I/O nodes that own OS resources (shared-memory segments, zmq sockets) must release
them in terminate(); a blocking node (Delay) must stay interruptible so terminate()
isn't stalled for the whole delay window.
"""
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from goofi.data import Data, DataType
from goofi.node import NodeEnv


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def test_sharedmemout_terminate_releases_segment():
    from goofi.nodes.outputs.sharedmemout import SharedMemOut

    name = "goofi-test-shm-hygiene"
    node = _standalone(SharedMemOut)
    node.setup()
    node.params.shared_memory.name.value = name
    node(data=Data(DataType.ARRAY, np.zeros(4, dtype=np.float32), {}))
    assert node.shm is not None

    node.terminate()

    assert node.shm is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name, create=False)  # segment unlinked


def test_zeromqout_terminate_closes_socket():
    import zmq

    node = _standalone(__import__("goofi.nodes.outputs.zeromqout", fromlist=["ZeroMQOut"]).ZeroMQOut)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.bind("inproc://goofi-test-zmqout")
    node.context = ctx
    node.socket = sock

    node.terminate()

    assert sock.closed


def test_delay_is_interruptible_when_not_alive():
    from goofi.nodes.signal.delay import Delay

    node = _standalone(Delay)
    node.params.delay.time.value = 2.0  # would block 2s if it slept unconditionally
    node._alive = False  # simulate terminate() having cleared the alive flag

    t0 = time.time()
    out = node(data=Data(DataType.ARRAY, np.zeros(3, dtype=np.float32), {}))
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"delay blocked {elapsed:.2f}s despite not being alive"
    assert out is not None
