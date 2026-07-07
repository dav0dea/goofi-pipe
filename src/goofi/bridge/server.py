"""aiohttp server hosting the SvelteKit SPA and bridge WebSockets.

Lives in the manager process. The HTTP server runs on its own thread
inside a private asyncio loop; the manager's main thread is free to
handle KeyboardInterrupt / sleep loops as before.

Public entry: `start_bridge(manager, host, port)` returns a `BridgeServer`
handle. Call `.shutdown()` from the manager on terminate.
"""
from __future__ import annotations

import asyncio
import errno
import importlib.resources as resources
import os
import shutil
import subprocess
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


# Build inputs whose mtime gates staleness: dirs are walked recursively, files
# stat'd directly. If any is newer than the emitted bundle, the SPA is rebuilt.
_BUILD_INPUTS = (
    "src",
    "static",
    "package.json",
    "package-lock.json",
    "svelte.config.js",
    "tsconfig.json",
    "vite.config.ts",
)


def _frontend_source_dir() -> Optional[Path]:
    """The repo `frontend/` source tree — only present in a dev checkout.

    Returns None for wheel installs (no source to build) and when
    `$GOOFI_FRONTEND_BUILD` points the static root somewhere we don't own.
    """
    if os.environ.get("GOOFI_FRONTEND_BUILD"):
        return None
    frontend = Path(__file__).resolve().parents[3] / "frontend"
    if (frontend / "package.json").is_file() and (frontend / "src").is_dir():
        return frontend
    return None


def _newest_input_mtime(frontend: Path) -> float:
    newest = 0.0
    for name in _BUILD_INPUTS:
        p = frontend / name
        if not p.exists():
            continue
        # Stat the input itself *and*, for a dir, every descendant. Including the
        # dir's own mtime catches a direct child being deleted/renamed — which
        # rglob alone would miss, since the removed file no longer shows up but
        # the parent dir's mtime bumps.
        items = [p, *p.rglob("*")] if p.is_dir() else [p]
        for item in items:
            try:
                newest = max(newest, item.stat().st_mtime)
            except OSError:
                pass
    return newest


def _frontend_is_stale(frontend: Path) -> bool:
    index = frontend / "build" / "index.html"
    if not index.is_file():
        return True
    try:
        built = index.stat().st_mtime
    except OSError:
        return True
    return _newest_input_mtime(frontend) > built


def maybe_build_frontend() -> None:
    """In a dev checkout, rebuild the SvelteKit SPA when its sources are newer
    than the emitted bundle, so `goofi-pipe` always serves the current frontend.

    No-op for wheel installs, when the bundle is already fresh, or when opted
    out via `$GOOFI_NO_FRONTEND_BUILD`. Missing tooling (`npm`/`node_modules`)
    or a failed build degrade to serving whatever bundle already exists.
    """
    if os.environ.get("GOOFI_NO_FRONTEND_BUILD"):
        return
    frontend = _frontend_source_dir()
    if frontend is None or not _frontend_is_stale(frontend):
        return

    npm = shutil.which("npm")
    if npm is None:
        print(
            "bridge: frontend bundle is stale but `npm` is not on PATH — serving the "
            "existing build. Run `npm run build` in frontend/ to refresh."
        )
        return
    if not (frontend / "node_modules").is_dir():
        print(
            "bridge: frontend bundle is stale but deps are missing — run `npm install` "
            "in frontend/. Serving the existing build."
        )
        return

    print("bridge: frontend sources changed — rebuilding (npm run build)…")
    try:
        subprocess.run([npm, "run", "build"], cwd=str(frontend), check=True)
        print("bridge: frontend rebuilt.")
    except subprocess.CalledProcessError as e:
        print(f"bridge: frontend build failed (exit {e.returncode}) — serving the previous build.")


def is_benign_disconnect(exc: object) -> bool:
    """True for a client that vanished mid-write (tab closed/reloaded). aiohttp's
    ClientConnectionResetError subclasses the builtin ConnectionResetError, so one
    isinstance check covers both — expected churn for a single-user local server,
    not a fault."""
    return isinstance(exc, ConnectionResetError)


def quiet_disconnect_exception_handler(loop, context: dict) -> None:
    """asyncio loop exception handler: demote a benign client disconnect to nothing.
    aiohttp schedules the permessage-deflate frame flush as a fire-and-forget task, so a
    client closing mid-broadcast otherwise surfaces as a noisy 'Task exception was never
    retrieved'. Every real error still falls through to the default handler."""
    if is_benign_disconnect(context.get("exception")):
        return
    loop.default_exception_handler(context)


class BridgeServer:
    """Daemon-thread aiohttp server orchestrating control + data hubs."""

    def __init__(self, manager, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._thread: Optional[threading.Thread] = None
        # `_started` means "startup resolved" — bound OR failed. `_startup_error`
        # carries the bind failure so start() can raise instead of hanging.
        self._started = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._url: Optional[str] = None

        self.control = ControlHub(self)
        self.data = DataHub(self)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="goofi-bridge", daemon=True)
        self._thread.start()
        # Wait for the server to actually bind (or fail) so the caller can print
        # the final URL without racing the listener — and surface a bind failure
        # instead of blocking the full timeout and then running forever with no UI.
        self._started.wait(timeout=10.0)
        if self._startup_error is not None:
            err = self._startup_error
            if isinstance(err, OSError) and err.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"port {self.port} is already in use — is another goofi-pipe running? "
                    "Pass --port to pick another."
                ) from err
            raise RuntimeError(f"bridge failed to start: {err}") from err
        if not self._started.is_set():
            raise RuntimeError("bridge did not start within 10s")

    def _install_exception_handler(self) -> None:
        """Quiet the benign client-disconnect noise on this bridge loop (see
        quiet_disconnect_exception_handler)."""
        self._loop.set_exception_handler(quiet_disconnect_exception_handler)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._install_exception_handler()
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            # Capture a startup (bind) failure so start() can raise it; a
            # post-bind crash lands here too but start() has already returned.
            self._startup_error = e
            print(f"bridge: server crashed: {e}")
        finally:
            # Unblock start()'s wait whether we bound, failed to bind, or exited —
            # `_started` is "startup resolved", not "bound". On the success path
            # _serve() already set it before entering its keep-alive loop.
            self._started.set()
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
                web.get("/data/{node}/{slot}/{kind}", self.data.handler),
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


def bind_warning(host: str) -> Optional[str]:
    """Return a security warning if `host` exposes the bridge beyond loopback.

    The bridge has NO authentication and nodes execute arbitrary Python
    (expression nodes), so a non-loopback bind is a remote-code-execution
    risk. Returns None for loopback hosts.
    """
    if host in ("127.0.0.1", "localhost", "::1"):
        return None
    return (
        f"WARNING: bridge bound to {host} — the UI is reachable from the network "
        "with NO authentication, and nodes execute arbitrary Python (expression "
        "nodes = remote code execution). Pass --bind 127.0.0.1 unless you intend this."
    )


def start_bridge(manager, host: str = "127.0.0.1", port: int = 8000) -> BridgeServer:
    """Spin up the bridge in a daemon thread and return the handle."""
    # Refresh the served bundle before binding so the printed URL lands on a
    # current frontend (dev checkout only; no-op once the build is fresh).
    maybe_build_frontend()
    warning = bind_warning(host)
    if warning:
        print(warning)
    bridge = BridgeServer(manager, host=host, port=port)
    bridge.start()
    return bridge
