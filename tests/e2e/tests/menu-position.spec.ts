import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

// Task 7's regression guard. `.panel-body` gains `container-type: inline-size` (the `@container`
// enablement). The add-node menu (`.menu-anchor`) is `position: fixed`, positioned in VIEWPORT
// coordinates (menuPos = window.innerWidth / clientX math), so it is portalled to <body>. Two
// independent checks, because either alone has a blind spot:
//   1. STRUCTURAL — the anchor really escaped the panel to <body>. This is the check with teeth
//      against the concrete regression (someone drops `use:portal`): `inline-size` containment does
//      NOT by itself trap fixed descendants in Chromium, so a position-only check would pass even
//      un-portalled today — but it WOULD break the moment the body gained a real containing-block
//      trigger (a transform/filter). Asserting the portal keeps the invariant that makes it safe.
//   2. POSITIONAL — the menu lands at its intended viewport point (not shifted by the panel offset),
//      which catches any real re-anchoring plus general placement bugs in openAddMenu.
//
// The panel-addressed path (`window.goofi`'s `openAddMenu`, which names a panel rather than a
// point) centres the menu over that editor: `openAddMenu(r.left + r.width/2, r.top + 60, 'center')`,
// where r is the editor root's viewport rect and the half-width comes from the MEASURED menu — so
// the expected x below is the centre minus half of the menu's own 320px box.
test('the add-node menu portals to <body> and opens at its intended viewport point', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// The editor root (`.canvas-wrap` is the rootEl openAddMenuCentered measures).
	const editor = page.locator('.canvas-wrap').first();
	await editor.waitFor();
	const r = (await editor.boundingBox())!;
	const expected = { x: r.x + r.width / 2 - 160, y: r.y + 60 };

	// Open the add-node menu through the panel-addressed façade command.
	await page.evaluate(() => (window as any).goofi.commands.openAddMenu());

	const anchor = page.getByTestId('add-node-menu-anchor');
	await expect(anchor, 'the menu is shown').toBeVisible();

	// 1. STRUCTURAL: the anchor escaped `.panel-body` — its parent is <body>, and no `.panel-body`
	//    ancestor remains. Dropping `use:portal` fails this immediately.
	const escaped = await anchor.evaluate((el) => ({
		parentIsBody: el.parentElement === document.body,
		insidePanelBody: !!el.closest('.panel-body')
	}));
	expect(escaped.parentIsBody, 'the menu anchor is portalled directly to <body>').toBe(true);
	expect(escaped.insidePanelBody, 'the menu anchor is not left inside a panel body').toBe(false);

	// 2. POSITIONAL: the rendered box sits at the viewport-relative target (a few px for borders).
	const box = (await anchor.boundingBox())!;
	expect(Math.abs(box.x - expected.x), 'menu left is at the viewport target').toBeLessThanOrEqual(2);
	expect(Math.abs(box.y - expected.y), 'menu top is at the viewport target').toBeLessThanOrEqual(2);
});

// D-M2's other half: the add-node menu must SHARE the `clampToViewport` SSOT, not re-derive it (or,
// on two of its paths, skip it). `Tab` in the active editor opens the 320px menu at the last
// window-level pointer position — so with the cursor near a far corner the menu rendered off-screen
// and the node could not be picked at all. RED before the four paths were routed through one
// measured, clamped `openAddMenu`.
test('Tab opens the add-node menu on-screen even with the pointer in the far corner', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const vp = page.viewportSize()!;

	// Make the editor the active panel (Tab is gated on it), then park the pointer in the corner —
	// `trackMouse` is a window listener, so this is the position `openMenuAtCursor` will use.
	await page.locator('.canvas-wrap').first().click({ position: { x: 20, y: 20 } });
	await page.mouse.move(vp.width - 3, vp.height - 3);
	await page.keyboard.press('Tab');

	const anchor = page.getByTestId('add-node-menu-anchor');
	await expect(anchor, 'Tab opened the menu').toBeVisible();
	const box = (await anchor.boundingBox())!;
	expect(box.x, 'the menu does not run off the left edge').toBeGreaterThanOrEqual(0);
	expect(box.y, 'the menu does not run off the top edge').toBeGreaterThanOrEqual(0);
	expect(box.x + box.width, 'the menu does not run off the right edge').toBeLessThanOrEqual(vp.width);
	expect(box.y + box.height, 'the menu does not run off the bottom edge').toBeLessThanOrEqual(
		vp.height
	);
	await page.keyboard.press('Escape');
	await expect(anchor).toBeHidden();
});

// The fourth entry point, and the only one whose geometry the unification restated: a TARGET port
// hangs the menu to its LEFT (the click point is the menu's RIGHT edge + 12px of clearance). That
// used to be a hardcoded `-332`; it is now the measured width, so this pins the resulting box.
test('clicking an input port opens the add-node menu to the LEFT of the port', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Buffer', 'signal', [700, 200]);
	await waitForNode(page, uid);

	// The backend graph is shared across this worker's specs, so the node comes back out even if an
	// assertion below throws — a leaked node breaks the fs-browser round trip further down the file.
	try {
		const port = page
			.locator(`.svelte-flow__node[data-id="${uid}"]`)
			.getByTestId('slot-input')
			.first();
		await expect(port, 'the node rendered its input port').toBeVisible();
		const portBox = (await port.boundingBox())!;
		await port.click();

		const anchor = page.getByTestId('add-node-menu-anchor');
		await expect(anchor, 'the port click opened the seeded menu').toBeVisible();
		const box = (await anchor.boundingBox())!;
		const clickX = portBox.x + portBox.width / 2;
		expect(box.x + box.width, 'the menu ends 12px left of the port').toBeCloseTo(clickX - 12, 0);
		expect(box.x, 'and still starts on-screen').toBeGreaterThanOrEqual(6);

		await page.keyboard.press('Escape');
		await expect(anchor).toBeHidden();
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});
