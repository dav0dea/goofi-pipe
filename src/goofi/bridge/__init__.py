"""HTTP + WebSocket bridge between the Python manager and the browser UI.

The bridge is the replacement for the dearpygui frontend. It runs inside
the manager process on a background asyncio loop (daemon thread), serves
the built SvelteKit SPA over HTTP, and exposes:

- `WS /control`: bidirectional JSON RPC + event stream for graph mutations,
  parameter updates, and patch persistence.
- `WS /data/<node>/<slot>`: per-subscription binary stream of encoded
  `Data` frames (GOOF wire format from `goofi.codec`).
"""
from goofi.bridge.server import BridgeServer, start_bridge

__all__ = ["BridgeServer", "start_bridge"]
