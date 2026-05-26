"""End-to-end Playwright tests against a live goofi-pipe bridge.

Fixtures spin up a real Manager (single-process for speed) on a free
port, point Chromium at it, and tear everything down afterward. The
SPA bundle is served by the bridge directly via `GOOFI_FRONTEND_BUILD`.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Generator, Iterator, List

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from goofi.manager import Manager

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_BUILD = REPO_ROOT / "frontend" / "build"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session", autouse=True)
def _frontend_build_env() -> None:
    if not FRONTEND_BUILD.exists():
        pytest.skip(
            f"frontend bundle missing at {FRONTEND_BUILD}; run "
            f"`(cd frontend && npm run build)` before running e2e tests"
        )
    os.environ.setdefault("GOOFI_FRONTEND_BUILD", str(FRONTEND_BUILD))


@pytest.fixture(scope="session")
def playwright_browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def bridge() -> Iterator[str]:
    """Boot a Manager on a fresh port, yield the URL, terminate after."""
    port = _free_port()
    mgr: List[Manager] = []

    def run():
        m = Manager(headless=False, use_multiprocessing=False, bridge_port=port, duration=120)
        mgr.append(m)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/api/healthz", timeout=0.2).read()
            break
        except Exception:
            time.sleep(0.05)
    else:
        raise RuntimeError(f"bridge {url} never bound")

    yield url

    if mgr:
        try:
            mgr[0].terminate()
        except Exception:
            pass


@pytest.fixture
def page(
    playwright_browser: Browser, bridge: str
) -> Iterator[Page]:
    ctx = playwright_browser.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    errors: List[str] = []

    def on_console(msg) -> None:
        if msg.type == "error":
            errors.append(f"console.{msg.type}: {msg.text}")

    page.on("console", on_console)
    page.on("pageerror", lambda err: errors.append(f"pageerror: {err}"))
    page.goto(bridge)
    page.wait_for_selector('[data-testid="topbar-add"]', timeout=10000)
    # expose errors list to tests via a custom attr
    setattr(page, "_console_errors", errors)
    yield page
    ctx.close()


def assert_no_console_errors(page: Page) -> None:
    errors = getattr(page, "_console_errors", [])
    assert not errors, f"unexpected console errors:\n" + "\n".join(errors)


@pytest.fixture
def shots(tmp_path_factory) -> Path:
    out = REPO_ROOT / "e2e" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    return out
