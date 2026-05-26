"""aiohttp server hosting the SvelteKit SPA and bridge WebSockets.

Lives in the manager process. The HTTP server runs on its own thread
inside a private asyncio loop; the manager's main thread is free to
handle KeyboardInterrupt / sleep loops as before.

Public entry: `start_bridge(manager, host, port)` returns a `BridgeServer`
handle. Call `.shutdown()` from the manager on terminate.
"""
from __future__ import annotations

import asyncio
import importlib.resources as resources
import threading
from pathlib import Path
from typing import Optional

from aiohttp import web

from goofi.bridge.control import ControlHub
from goofi.bridge.data import DataHub


def _resolve_static_root() -> Optional[Path]:
    """Find the built SvelteKit SPA, if any.

    Search order:
    1. `$GOOFI_FRONTEND_BUILD` (env var override)
    2. repo `frontend/build` (development checkout)
    3. installed package data `goofi/web` (wheel install)
    """
    import os

    env = os.environ.get("GOOFI_FRONTEND_BUILD")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # Repo checkout — `src/goofi/bridge/server.py` → `frontend/build`
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    cand = repo_root / "frontend" / "build"
    if cand.exists():
        return cand

    # Installed package: shipped under `goofi.web` data dir.
    try:
        with resources.as_file(resources.files("goofi").joinpath("web")) as p:
            if p.exists():
                return Path(p)
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return None


class BridgeServer:
    """Daemon-thread aiohttp server orchestrating control + data hubs."""

    def __init__(self, manager, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._url: Optional[str] = None

        self.control = ControlHub(self)
        self.data = DataHub(self)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="goofi-bridge", daemon=True)
        self._thread.start()
        # Wait for the server to actually bind so the caller can print the
        # final URL without racing the listener.
        self._started.wait(timeout=10.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"bridge: server crashed: {e}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        # Disable HTTP caching for every response — the manager is a single-
        # user local dev server, and serving stale JS/CSS during a `npm run
        # build` cycle is far worse than the negligible cost of re-fetching.
        @web.middleware
        async def no_cache(request: web.Request, handler):
            resp = await handler(request)
            if isinstance(resp, web.StreamResponse):
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
            return resp

        app.middlewares.append(no_cache)
        app.add_routes(
            [
                web.get("/control", self.control.handler),
                web.get("/data/{node}/{slot}", self.data.handler),
                web.get("/api/healthz", self._healthz),
            ]
        )

        static_root = _resolve_static_root()
        if static_root is not None:
            # SPA fallback: anything that isn't a static file or API → index.html.
            async def spa_handler(request: web.Request) -> web.StreamResponse:
                rel = request.match_info.get("path", "")
                target = static_root / rel if rel else static_root / "index.html"
                if target.is_file():
                    return web.FileResponse(target)
                return web.FileResponse(static_root / "index.html")

            app.router.add_get("/", spa_handler)
            app.router.add_get("/{path:.*}", spa_handler)
        else:
            async def placeholder(request: web.Request) -> web.Response:
                return web.Response(
                    text=(
                        "<h1>goofi-pipe bridge</h1>"
                        "<p>The frontend bundle was not found. Run <code>npm run build</code> "
                        "in <code>frontend/</code>, or start <code>npm run dev</code> "
                        "(it proxies <code>/control</code> + <code>/data</code> here).</p>"
                    ),
                    content_type="text/html",
                )

            app.router.add_get("/", placeholder)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self._url = f"http://{self.host}:{self.port}"
        self._started.set()
        # Keep the loop alive until shutdown() cancels the task.
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    @property
    def url(self) -> Optional[str]:
        return self._url

    def shutdown(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        async def _stop():
            try:
                await self.control.broadcast({"event": "manager_shutdown", "payload": {}})
            except Exception:
                pass
            await self.control.close_all()
            await self.data.close_all()
            if self._runner is not None:
                await self._runner.cleanup()
            # Cancel everything else so run_until_complete returns.
            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()

        try:
            asyncio.run_coroutine_threadsafe(_stop(), loop).result(timeout=5.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def schedule(self, coro) -> None:
        """Schedule `coro` on the bridge loop from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(coro, loop)

    @staticmethod
    async def _healthz(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})


def start_bridge(manager, host: str = "127.0.0.1", port: int = 8000) -> BridgeServer:
    """Spin up the bridge in a daemon thread and return the handle."""
    bridge = BridgeServer(manager, host=host, port=port)
    bridge.start()
    return bridge
