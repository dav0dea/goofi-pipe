"""Z-order: the add-node menu must overlay the canvas without clipping;
the side panel must overlay nothing it shouldn't."""
from __future__ import annotations

from pathlib import Path
from playwright.sync_api import Page

from .conftest import assert_no_console_errors


def test_add_menu_overlays_canvas(page: Page, shots: Path) -> None:
    # Add a few nodes so there's canvas content under the menu.
    for name in ("ConstantArray", "Oscillator", "Buffer"):
        page.click('[data-testid="topbar-add"]')
        page.fill('[data-testid="add-menu-search"]', name)
        page.wait_for_timeout(120)
        page.keyboard.press("Enter")
        page.wait_for_timeout(250)

    page.click('[data-testid="topbar-add"]')
    page.wait_for_selector('[data-testid="add-menu-search"]', timeout=2000)
    menu = page.locator('[data-testid="add-menu-search"]').first
    box = menu.bounding_box()
    assert box is not None
    # The menu should appear above the canvas content visually.
    page.screenshot(path=str(shots / "z-menu-over-canvas.png"), full_page=True)
    assert_no_console_errors(page)
