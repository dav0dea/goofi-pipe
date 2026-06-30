"""A client that closes a WebSocket mid-broadcast (a tab reload/close) makes aiohttp's
fire-and-forget compressed-frame flush task fail with ClientConnectionResetError, which
asyncio then logs as a noisy 'Task exception was never retrieved'. That is expected churn
for a single-user local server, not a fault — the bridge loop installs an exception
handler that demotes it while letting every real error through.
"""
import asyncio

from aiohttp.client_exceptions import ClientConnectionResetError

from goofi.bridge.server import (
    BridgeServer,
    is_benign_disconnect,
    quiet_disconnect_exception_handler,
)


def test_is_benign_disconnect_covers_connection_reset():
    assert is_benign_disconnect(ConnectionResetError()) is True
    assert is_benign_disconnect(ClientConnectionResetError()) is True  # aiohttp subclass
    assert is_benign_disconnect(ValueError("real bug")) is False
    assert is_benign_disconnect(None) is False


def test_handler_swallows_benign_and_delegates_others():
    delegated = []

    class FakeLoop:
        def default_exception_handler(self, ctx):
            delegated.append(ctx)

    loop = FakeLoop()
    quiet_disconnect_exception_handler(loop, {"exception": ClientConnectionResetError()})
    assert delegated == []  # benign disconnect is swallowed
    quiet_disconnect_exception_handler(loop, {"message": "boom", "exception": ValueError("x")})
    assert len(delegated) == 1  # a real error falls through to the default handler


def test_server_installs_quiet_handler():
    srv = BridgeServer.__new__(BridgeServer)
    srv._loop = asyncio.new_event_loop()
    try:
        srv._install_exception_handler()
        assert srv._loop.get_exception_handler() is quiet_disconnect_exception_handler
    finally:
        srv._loop.close()
