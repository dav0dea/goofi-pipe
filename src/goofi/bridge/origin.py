"""Cross-origin guard for the bridge WebSocket planes (no-auth, local/trusted-LAN).

The /control and /data planes have no authentication and nodes execute arbitrary
Python (expression nodes). Without an Origin check, any web page the user happens
to visit could open ``ws://127.0.0.1:8000/control`` and drive those nodes —
cross-site WebSocket hijacking / DNS-rebinding, i.e. drive-by remote code
execution on the operator's own machine.

This rejects WS upgrades whose Origin is a foreign site while staying invisible to
legitimate use: the local SPA (including the vite dev proxy on another port),
direct loopback access, an explicit LAN bind, and native non-browser clients all
still connect. It adds no authentication — it only ties browser connections to the
origin that served the app.
"""
from typing import Optional
from urllib.parse import urlsplit

from aiohttp import web

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _hostname(value: str) -> Optional[str]:
    # urlsplit only populates .hostname when a netloc is present: an Origin
    # ("scheme://host:port") has one, a bare Host header ("host:port") does not —
    # so prefix "//" when the scheme/netloc separator is absent.
    candidate = value if "//" in value else f"//{value}"
    try:
        return urlsplit(candidate).hostname
    except ValueError:
        return None


def origin_allowed(origin: Optional[str], host_header: Optional[str], bind_host: str) -> bool:
    """Whether a WS upgrade carrying this ``Origin`` may connect.

    Allowed when any of:
    - no Origin (a native, non-browser client — nothing to forge);
    - the Origin hostname is loopback (the local SPA + the dev proxy on :5173);
    - it equals the bind host (a page served from an explicit ``--bind`` LAN IP);
    - it matches the request ``Host`` (same-origin in the general case).
    A foreign site (e.g. ``evil.com``) matches none and is rejected.
    """
    if not origin:
        return True
    origin_host = _hostname(origin)
    if origin_host is None:
        return False
    if origin_host in _LOOPBACK:
        return True
    if origin_host == _hostname(bind_host):
        return True
    if host_header and origin_host == _hostname(host_header):
        return True
    return False


def reject_cross_origin(request: web.Request, bind_host: str) -> Optional[web.Response]:
    """A 403 response for a foreign-Origin WS upgrade, or None to proceed — the single
    guard the /control and /data handlers both apply at their door (each keeps its own
    comment explaining that plane's specific threat)."""
    if origin_allowed(request.headers.get("Origin"), request.headers.get("Host"), bind_host):
        return None
    return web.Response(status=403, text="cross-origin WebSocket rejected")
