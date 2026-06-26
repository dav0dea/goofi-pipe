"""Cross-origin guard for the no-auth bridge WS planes (CSWSH / drive-by RCE)."""
from goofi.bridge.origin import origin_allowed


def test_absent_origin_is_allowed():
    # Native (non-browser) clients — the e2e harness, python ws clients — send no
    # Origin; there is nothing to forge, so they connect.
    assert origin_allowed(None, "127.0.0.1:8000", "127.0.0.1") is True
    assert origin_allowed("", "127.0.0.1:8000", "127.0.0.1") is True


def test_loopback_origins_allowed_including_dev_proxy():
    # Local SPA (any loopback host/port) + the vite dev proxy on :5173.
    assert origin_allowed("http://localhost:8000", "localhost:8000", "127.0.0.1") is True
    assert origin_allowed("http://127.0.0.1:8000", "127.0.0.1:8000", "127.0.0.1") is True
    assert origin_allowed("http://localhost:5173", "127.0.0.1:8000", "127.0.0.1") is True
    assert origin_allowed("http://[::1]:8000", "[::1]:8000", "127.0.0.1") is True


def test_lan_bind_host_origin_allowed():
    # Explicit --bind to a LAN IP: the page served from that IP may connect.
    assert origin_allowed("http://192.168.1.5:8000", "192.168.1.5:8000", "192.168.1.5") is True


def test_same_origin_via_host_header_allowed():
    # General same-origin: Origin host equals the Host the browser targeted.
    assert origin_allowed("http://my-host.local:8000", "my-host.local:8000", "0.0.0.0") is True


def test_foreign_origin_rejected():
    # A malicious page the user visits cannot drive the loopback control plane.
    assert origin_allowed("http://evil.com", "127.0.0.1:8000", "127.0.0.1") is False
    assert origin_allowed("https://evil.com:8000", "127.0.0.1:8000", "127.0.0.1") is False
    # A look-alike that is neither loopback, bind host, nor the targeted Host.
    assert origin_allowed("http://192.168.1.99:8000", "127.0.0.1:8000", "127.0.0.1") is False


def test_malformed_origin_rejected():
    assert origin_allowed("not a url", "127.0.0.1:8000", "127.0.0.1") is False


# ---- integration: the guard is actually wired into the live /control route ----

def test_control_route_returns_403_for_foreign_origin():
    """Drive the real ControlHub.handler through an aiohttp test server: a foreign
    Origin gets 403 BEFORE any upgrade; a loopback Origin passes the guard (and
    then fails the WS handshake with a non-403, since it's a plain GET)."""
    import asyncio
    from types import SimpleNamespace

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from goofi.bridge.control import ControlHub

    async def run():
        hub = ControlHub(SimpleNamespace(host="127.0.0.1", manager=None))
        app = web.Application()
        app.add_routes([web.get("/control", hub.handler)])
        async with TestClient(TestServer(app)) as client:
            foreign = await client.get("/control", headers={"Origin": "http://evil.com"})
            assert foreign.status == 403  # rejected at the door
            ok = await client.get("/control", headers={"Origin": "http://localhost:5173"})
            assert ok.status != 403  # guard passed; WS handshake fails for another reason

    asyncio.new_event_loop().run_until_complete(run())
