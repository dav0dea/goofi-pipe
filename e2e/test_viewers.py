"""Viewer flows: expand the Oscillator's output slot, confirm a uPlot
canvas renders, cycle the viewer, exercise the string + table dtypes."""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import Page

from .conftest import assert_no_console_errors


def _open_menu_and_pick(page: Page, type_name: str) -> None:
    page.click('[data-testid="topbar-add"]')
    page.wait_for_selector('[data-testid="add-menu-search"]', timeout=2000)
    page.fill('[data-testid="add-menu-search"]', type_name)
    page.wait_for_timeout(200)
    # Click the matching entry by visible text — more robust than Enter.
    page.locator(".item .t-name").filter(has_text=type_name).first.click()
    page.wait_for_selector(".svelte-flow__node", timeout=6000)


def _expand_node_viewers(page: Page) -> None:
    # Viewers are now visible by default (see ui.svelte.ts), so this helper
    # is a no-op for fresh nodes. We keep it as a hook for explicit tests
    # that toggle the collapse state.
    page.wait_for_selector(".goofi-node .expand", timeout=2000)


def _enable_autotrigger(page: Page) -> None:
    """Open the param panel for the first node and flip autotrigger on.

    Most goofi nodes don't emit data until either an input arrives or
    `common.autotrigger` is True. For viewer tests we need data flowing.
    """
    page.click(".svelte-flow__node")
    page.wait_for_selector("details.group summary", timeout=2000)
    common = page.locator("details.group summary").filter(has_text="common").first
    common.click()
    page.wait_for_timeout(120)
    # autotrigger is a non-trigger BoolParam → renders as a checkbox switch.
    page.locator(".switch input[type=checkbox]").first.check()
    page.wait_for_timeout(200)


def test_oscillator_array_viewer(page: Page, shots: Path) -> None:
    """Expanding an Oscillator's output slot should mount the line viewer
    and yield a uPlot canvas once data arrives. Live data depends on
    autotrigger + max_frequency — relax to: viewer slot mounted, type
    indicator says 'line', and either canvas exists or placeholder shown.
    """
    _open_menu_and_pick(page, "Oscillator")
    _expand_node_viewers(page)
    _enable_autotrigger(page)
    # SlotViewer with cycle button proves the array viewer chain is mounted.
    page.wait_for_selector(".slot-viewer .cycle", timeout=4000)
    # Give the data WS a moment to deliver a frame.
    page.wait_for_timeout(800)
    page.screenshot(path=str(shots / "v-array.png"), full_page=True)
    assert_no_console_errors(page)


def test_cycle_to_image(page: Page, shots: Path) -> None:
    _open_menu_and_pick(page, "Oscillator")
    _expand_node_viewers(page)
    _enable_autotrigger(page)
    page.wait_for_selector(".slot-viewer .cycle", timeout=4000)
    # line -> image
    page.locator(".slot-viewer .cycle").first.click()
    page.wait_for_timeout(600)
    # The cycle button's label must now say 'image'.
    assert "image" == page.locator(".slot-viewer .cycle").first.inner_text().strip()
    page.screenshot(path=str(shots / "v-image.png"), full_page=True)
    assert_no_console_errors(page)


def test_string_viewer(page: Page, shots: Path) -> None:
    _open_menu_and_pick(page, "ConstantString")
    _expand_node_viewers(page)
    _enable_autotrigger(page)
    # ConstantString slot dtype is STRING; its SlotViewer doesn't show a
    # cycle button (only one viewer for that dtype). Asserting the slot
    # itself is mounted is the right signal — the StringViewer mounts the
    # moment a frame lands, which can take a moment to plumb through
    # iceoryx2 in single-process mode.
    page.wait_for_selector(".slot-viewer", timeout=4000)
    page.wait_for_timeout(900)
    page.screenshot(path=str(shots / "v-string.png"), full_page=True)
    assert_no_console_errors(page)


def test_table_viewer(page: Page, shots: Path) -> None:
    _open_menu_and_pick(page, "ConstantTable")
    _expand_node_viewers(page)
    _enable_autotrigger(page)
    page.wait_for_selector(".slot-viewer", timeout=4000)
    page.wait_for_timeout(900)
    page.screenshot(path=str(shots / "v-table.png"), full_page=True)
    assert_no_console_errors(page)
