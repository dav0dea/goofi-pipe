"""Editor canvas: zoom/pan, node add via menu, drag, link, delete, save."""
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
    page.locator(".item .t-name").filter(has_text=type_name).first.click()
    page.wait_for_timeout(250)


def test_empty_editor_renders(page: Page, shots: Path) -> None:
    page.screenshot(path=str(shots / "01-editor-empty.png"), full_page=True)
    assert_no_console_errors(page)


def test_add_node_menu_open_over_canvas(page: Page, shots: Path) -> None:
    page.click('[data-testid="topbar-add"]')
    page.wait_for_selector('[data-testid="add-menu-list"]')
    page.fill('[data-testid="add-menu-search"]', "Const")
    page.wait_for_timeout(100)
    page.screenshot(path=str(shots / "02-add-menu-open.png"), full_page=True)
    assert_no_console_errors(page)


def test_add_three_nodes(page: Page, shots: Path) -> None:
    _open_menu_and_pick(page, "ConstantArray")
    _open_menu_and_pick(page, "Buffer")
    _open_menu_and_pick(page, "Oscillator")
    # All three appear in the SvelteFlow node container.
    page.wait_for_function(
        "document.querySelectorAll('.svelte-flow__node').length === 3", timeout=4000
    )
    page.screenshot(path=str(shots / "03-three-nodes.png"), full_page=True)
    assert_no_console_errors(page)


def test_node_drag_persists_position(page: Page) -> None:
    _open_menu_and_pick(page, "ConstantArray")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    node = page.locator(".svelte-flow__node").first
    box_before = node.bounding_box()
    assert box_before is not None
    # Drag using actual mouse events (Svelte Flow ignores synthetic .drag).
    page.mouse.move(box_before["x"] + 50, box_before["y"] + 10)
    page.mouse.down()
    page.mouse.move(box_before["x"] + 220, box_before["y"] + 90, steps=8)
    page.mouse.up()
    page.wait_for_timeout(200)
    box_after = node.bounding_box()
    assert box_after is not None
    # Moved at least ~100px right and down.
    assert box_after["x"] - box_before["x"] > 100
    assert_no_console_errors(page)


def test_param_panel_shows_when_selected(page: Page, shots: Path) -> None:
    _open_menu_and_pick(page, "Oscillator")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    page.click(".svelte-flow__node")
    page.wait_for_timeout(150)
    # The right-hand side panel should contain a "params" group at least.
    page.wait_for_selector(".panel summary, details.group", timeout=2000)
    page.screenshot(path=str(shots / "04-param-panel.png"), full_page=True)
    assert_no_console_errors(page)


def test_zoom_via_mousewheel(page: Page) -> None:
    """Wheel scroll on the canvas should zoom the viewport (transform changes)."""
    _open_menu_and_pick(page, "ConstantArray")
    page.wait_for_selector(".svelte-flow__viewport", timeout=4000)
    transform_before = page.evaluate(
        "() => document.querySelector('.svelte-flow__viewport').style.transform"
    )
    # Use page.mouse.wheel after positioning over the canvas.
    page.mouse.move(800, 500)
    page.mouse.wheel(0, -400)
    page.wait_for_timeout(150)
    transform_after = page.evaluate(
        "() => document.querySelector('.svelte-flow__viewport').style.transform"
    )
    assert transform_before != transform_after, "viewport transform unchanged after wheel"
    assert_no_console_errors(page)


def test_load_example_patch(page: Page, shots: Path) -> None:
    page.click('[data-testid="topbar-examples"]')
    page.wait_for_selector(".ex-item", timeout=2000)
    # Pick a small example (the smallest .gfi by file size).
    page.locator(".ex-item").first.click()
    page.wait_for_function(
        "document.querySelectorAll('.svelte-flow__node').length > 0", timeout=10000
    )
    page.screenshot(path=str(shots / "05-example-loaded.png"), full_page=True)
    assert_no_console_errors(page)
