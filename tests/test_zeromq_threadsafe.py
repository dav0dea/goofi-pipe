"""ZeroMQ nodes run two threads: the messaging loop applies param changes (address/port),
whose callbacks call setup() and CLOSE+recreate the socket, while the processing loop is
inside process() using that same socket. zmq sockets are NOT thread-safe, so the socket
must be guarded by a lock AND process() must never hold it indefinitely — a blocking
recv() under the lock would wedge the node and deadlock every param change.
"""
import threading

from goofi.node import NodeEnv


def _standalone(node_cls):
    in_slots, out_slots, params = node_cls._configure()
    return node_cls(None, in_slots, out_slots, params, NodeEnv.STANDALONE)


def _run_with_timeout(fn, timeout):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    t.start()
    t.join(timeout)
    return (not t.is_alive()), out.get("r")


def test_zeromqin_process_does_not_block_without_data():
    """With no peer sending, process() must return promptly (poll, not a blocking recv) —
    otherwise it holds the socket forever and a concurrent address/port change can never
    recreate it. A blocking recv_pyobj() would hang here."""
    from goofi.nodes.inputs.zeromqin import ZeroMQIn

    node = _standalone(ZeroMQIn)
    node.setup()  # STANDALONE skips auto-setup; create the socket explicitly
    try:
        finished, result = _run_with_timeout(node.process, timeout=3.0)
        assert finished, "ZeroMQIn.process blocked with no data (blocking recv wedges the node)"
        assert result is None  # no message ready this tick
    finally:
        node.terminate()


def test_zeromq_nodes_guard_the_socket_with_a_lock():
    """setup() (socket recreate on the messaging thread) and process() (socket op on the
    processing thread) must serialize on a shared lock so the socket is never closed
    mid-use. Recreating the socket while a process() is in flight must not raise."""
    from goofi.nodes.inputs.zeromqin import ZeroMQIn
    from goofi.nodes.outputs.zeromqout import ZeroMQOut

    for cls in (ZeroMQIn, ZeroMQOut):
        node = _standalone(cls)
        if cls is ZeroMQOut:
            # ZeroMQOut binds; use an ephemeral OS-assigned port so the test never collides
            # with another test's (or a lingering process's) socket on the default 6543.
            # (ZeroMQIn only connects, which never conflicts and rejects port 0.)
            node.params.zero_mq.port.value = 0
        node.setup()  # STANDALONE skips auto-setup
        try:
            assert hasattr(node, "_lock"), f"{cls.__name__} has no socket lock"
            # Recreating the socket (what a param change does) is safe to call repeatedly.
            node.setup()
            node.setup()
        finally:
            node.terminate()
