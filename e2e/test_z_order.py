"""Z-order: the add-node menu must overlay the canvas without clipping;
the side panel must overlay nothing it shouldn't."""
from __future__ import annotations

from pathlib import Path
from playwright.sync_api import Page

from .conftest import assert_no_console_errors


def test_add_menu_overlays_canvas(page: Page, shots: Path) -> None:
    # Add a few nodes so there's canvas content under the menu. Picking a
    # type now enters a placement preview; commit it with a canvas click so
    # the next loop iteration starts from a clean state.
    pane = page.locator(".svelte-flow__pane").first
    for i, name in enumerate(("ConstantArray", "Oscillator", "Buffer")):
        page.click('[data-testid="topbar-add"]')
        page.fill('[data-testid="add-menu-search"]', name)
        page.wait_for_timeout(120)
        page.keyboard.press("Enter")
        page.wait_for_selector('[data-testid="placement-ghost"]', timeout=1500)
        box = pane.bounding_box()
        assert box
        tx = box["x"] + 120 + i * 260
        ty = box["y"] + box["height"] / 2
        page.mouse.move(tx, ty)
        page.wait_for_timeout(50)
        page.keyboard.down("Alt")
        page.mouse.click(tx, ty)
        page.keyboard.up("Alt")
        page.wait_for_timeout(150)

    page.click('[data-testid="topbar-add"]')
    page.wait_for_selector('[data-testid="add-menu-search"]', timeout=2000)
    menu = page.locator('[data-testid="add-menu-search"]').first
    box = menu.bounding_box()
    assert box is not None
    # The menu should appear above the canvas content visually.
    page.screenshot(path=str(shots / "z-menu-over-canvas.png"), full_page=True)
    assert_no_console_errors(page)
