"""BridgeServer startup must surface a bind failure, not hang for 10s and then
run forever with no UI (a second goofi-pipe on the same port)."""
import socket
import time
from types import SimpleNamespace

import pytest

from goofi.bridge.server import BridgeServer


def _fake_manager():
    # start() only binds the socket; the hubs aren't exercised without a request.
    return SimpleNamespace(instance_id="test", save_path=None, unsaved_changes=False, layout=None)


def test_start_raises_promptly_on_port_in_use():
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    port = occupied.getsockname()[1]

    server = BridgeServer(_fake_manager(), host="127.0.0.1", port=port)
    try:
        t0 = time.time()
        with pytest.raises(RuntimeError) as ei:
            server.start()
        assert time.time() - t0 < 5.0, "start() blocked on the 10s backstop instead of failing fast"
        assert str(port) in str(ei.value)
    finally:
        server.shutdown()
        occupied.close()


def test_start_succeeds_on_a_free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = BridgeServer(_fake_manager(), host="127.0.0.1", port=port)
    try:
        server.start()  # must not raise
        assert server.url == f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
