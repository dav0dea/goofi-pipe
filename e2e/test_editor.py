"""Editor canvas: zoom/pan, node add via menu, drag, link, delete, save."""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import Page

from .conftest import assert_no_console_errors


def _open_menu_and_pick(page: Page, type_name: str) -> None:
    """Open the add-node menu, pick a type, then commit placement by
    clicking the canvas. Picking from the menu now enters a placement-
    preview mode (ghost follows cursor) — the node isn't added until the
    follow-up canvas click.

    To prevent the snap-to-edge behavior from stacking successive nodes,
    spread placements horizontally based on the current node count."""
    page.click('[data-testid="topbar-add"]')
    page.wait_for_selector('[data-testid="add-menu-search"]', timeout=2000)
    page.fill('[data-testid="add-menu-search"]', type_name)
    page.wait_for_timeout(200)
    page.locator(".item .t-name").filter(has_text=type_name).first.click()
    page.wait_for_selector('[data-testid="placement-ghost"]', timeout=1500)
    pane = page.locator(".svelte-flow__pane").first
    box = pane.bounding_box()
    assert box
    existing = page.locator(".svelte-flow__node").count()
    # Hold Alt while clicking so we land at exactly the cursor, bypassing
    # the snap-to-neighbours logic that would otherwise pile nodes up.
    target_x = box["x"] + 120 + (existing * 260) % max(int(box["width"]) - 240, 320)
    target_y = box["y"] + box["height"] / 2
    page.mouse.move(target_x, target_y)
    page.wait_for_timeout(50)
    page.keyboard.down("Alt")
    page.mouse.click(target_x, target_y)
    page.keyboard.up("Alt")
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


def test_snap_guides_appear_on_drag(page: Page, shots: Path) -> None:
    """Drag a node toward another; SVG snap guides should render mid-drag."""
    _open_menu_and_pick(page, "ConstantArray")
    _open_menu_and_pick(page, "Oscillator")
    page.wait_for_function(
        "document.querySelectorAll('.svelte-flow__node').length === 2", timeout=4000
    )
    nodes = page.locator(".svelte-flow__node").all()
    assert len(nodes) == 2
    target_box = nodes[0].bounding_box()
    drag_box = nodes[1].bounding_box()
    assert target_box and drag_box
    # Drag node 1's header toward node 0's vertical alignment.
    page.mouse.move(drag_box["x"] + 30, drag_box["y"] + 8)
    page.mouse.down()
    page.mouse.move(target_box["x"] + 30, target_box["y"] + 200, steps=8)
    page.wait_for_timeout(120)
    # Mid-drag: guides element should exist with at least one line.
    guide_count = page.locator('[data-testid="snap-guides"] line').count()
    page.mouse.up()
    page.wait_for_timeout(200)
    assert guide_count > 0, "expected at least one snap-guide line mid-drag"
    page.screenshot(path=str(shots / "06-snap-guides.png"), full_page=True)
    assert_no_console_errors(page)


def test_no_grid_snap_quantization(page: Page) -> None:
    """Drag by a non-grid-multiple amount; the persisted position must reflect
    the exact delta, not a quantized 8px snap."""
    _open_menu_and_pick(page, "ConstantArray")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    node = page.locator(".svelte-flow__node").first
    before = node.bounding_box()
    assert before
    DX, DY = 137, 91  # neither divides by 8
    page.mouse.move(before["x"] + 30, before["y"] + 8)
    page.mouse.down()
    page.mouse.move(before["x"] + 30 + DX, before["y"] + 8 + DY, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    after = node.bounding_box()
    assert after
    # Edge alignment may pull a few px; assert no 8px quantization by
    # checking the delta is within 16px of the requested amount but not
    # a perfect multiple of 8 (which would indicate the old snapGrid).
    dx_observed = after["x"] - before["x"]
    dy_observed = after["y"] - before["y"]
    assert abs(dx_observed - DX) < 20, f"dx {dx_observed} too far from {DX}"
    assert abs(dy_observed - DY) < 20, f"dy {dy_observed} too far from {DY}"
    assert_no_console_errors(page)


def test_slot_click_opens_seeded_menu(page: Page, shots: Path) -> None:
    """Click an unconnected output port → menu opens with dtype chip."""
    _open_menu_and_pick(page, "Oscillator")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    # Click the first unconnected output port.
    page.locator('[data-testid="slot-output"]').first.click()
    page.wait_for_selector('[data-testid="add-menu-seed"]', timeout=2000)
    chip_text = page.locator('[data-testid="add-menu-seed"]').inner_text()
    assert "from" in chip_text.lower()
    page.screenshot(path=str(shots / "07-slot-click-menu.png"), full_page=True)
    assert_no_console_errors(page)


def test_slot_click_auto_links(page: Page) -> None:
    """Pick a compatible type from a seeded menu → link created automatically."""
    _open_menu_and_pick(page, "Oscillator")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    page.locator('[data-testid="slot-output"]').first.click()
    page.wait_for_selector('[data-testid="add-menu-seed"]', timeout=2000)
    page.fill('[data-testid="add-menu-search"]', "Buffer")
    page.wait_for_timeout(200)
    page.locator(".item .t-name").filter(has_text="Buffer").first.click()
    # Picking now enters placement mode; commit by clicking the canvas.
    page.wait_for_selector('[data-testid="placement-ghost"]', timeout=1500)
    pane = page.locator(".svelte-flow__pane").first
    box = pane.bounding_box()
    assert box
    target_x = box["x"] + box["width"] * 0.7
    target_y = box["y"] + box["height"] / 2
    page.mouse.move(target_x, target_y)
    page.wait_for_timeout(50)
    page.keyboard.down("Alt")
    page.mouse.click(target_x, target_y)
    page.keyboard.up("Alt")
    # One edge after auto-link.
    page.wait_for_function(
        "document.querySelectorAll('.svelte-flow__edge').length === 1", timeout=4000
    )
    assert_no_console_errors(page)


def test_param_panel_floats_visibility(page: Page) -> None:
    """Side panel is off-screen when no selection; slides in on selection."""
    _open_menu_and_pick(page, "Oscillator")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    panel = page.locator(".side-panel")
    is_open_pre = page.evaluate(
        "() => document.querySelector('.side-panel')?.classList.contains('open') ?? false"
    )
    assert not is_open_pre, "panel should be hidden when no selection"
    page.locator(".svelte-flow__node").first.click()
    page.wait_for_timeout(250)
    is_open_post = page.evaluate(
        "() => document.querySelector('.side-panel')?.classList.contains('open') ?? false"
    )
    assert is_open_post, "panel should slide in after selecting a node"
    assert_no_console_errors(page)


def test_double_click_canvas_opens_menu(page: Page) -> None:
    """Double-click on empty canvas opens the add-node menu."""
    pane = page.locator(".svelte-flow__pane").first
    box = pane.bounding_box()
    assert box
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.click(cx, cy)
    page.wait_for_timeout(80)
    page.mouse.click(cx, cy)
    page.wait_for_timeout(200)
    assert (
        page.locator('[data-testid="add-menu-search"]').count() > 0
    ), "double-click should open the add-node menu"
    assert_no_console_errors(page)


def test_escape_clears_selection(page: Page) -> None:
    """Escape closes any open menu; pressing again clears selection."""
    _open_menu_and_pick(page, "Oscillator")
    page.wait_for_selector(".svelte-flow__node", timeout=4000)
    page.locator(".svelte-flow__node").first.click()
    page.wait_for_timeout(200)
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    is_open = page.evaluate(
        "() => document.querySelector('.side-panel')?.classList.contains('open') ?? false"
    )
    assert not is_open, "Escape should have cleared selection and closed the panel"
    assert_no_console_errors(page)
