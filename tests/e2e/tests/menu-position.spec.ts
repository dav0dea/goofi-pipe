import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

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
// `openAddMenu` centres the menu over the active editor: menuPos = { x: r.left + r.width/2 - 160,
// y: r.top + 60 } where r is the editor root's viewport rect.
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

	// Open the add-node menu through the same façade the TopBar "Add node" button uses.
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
