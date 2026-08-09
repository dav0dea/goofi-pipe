import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import { nodes, nodeParams } from '../lib/goofi';
import { openAddMenuByPress, paletteItem } from '../lib/placement';
import { settledBar } from '../lib/topbar';

/**
 * **The headline test for sub-project R.** Its §5 success criterion, verbatim: at 412px portrait a
 * user can *add a node, connect it, open its parameters, change one, and save*. Before R the first
 * and the last were impossible — the TopBar's intrinsic minimum was ≈628px, so Save, Load and the
 * add-node door were all off-screen, and the layout tab strip was squeezed to nothing beside them.
 *
 * Every step goes through the REAL touch surface: a long press to open the add-node menu, taps to
 * pick and place, a tap on a connector pill to seed the next node (which is how a phone makes a
 * cable — dragging one is a fine-pointer gesture R deliberately does not re-implement), the
 * inspector's own rendered control for the param, and the app header's own Save button for the save.
 * The `window.goofi` façade is used only to READ the result back.
 *
 * Hermeticity: the save lands in a per-run temp directory removed in afterAll, and the test hands
 * the shared backend back unnamed and empty — `resetPatch` (a manager-side `new`) is what resets
 * `save_path` to null, which later specs assume when they click Save expecting the browser.
 */

let scratch = '';
const patchName = `r-touch-${process.pid}-${Date.now()}`;

test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-touch-')));
});
test.afterAll(() => {
	fs.rmSync(scratch, { recursive: true, force: true });
});

/** Hand the shared backend back empty AND unnamed even when the test fails partway — a leftover
 * node changes where the NEXT spec's `emptySpot` may press, and a leftover NAME turns its Save
 * into a silent overwrite. The journey saves the patch, so both are reachable failure states. */
test.afterEach(async ({ page }) => {
	await page.evaluate(() => (window as any).goofi.commands.newPatch()).catch(() => {});
});

function links(page: Page): Promise<Array<Record<string, string>>> {
	return page.evaluate(() => (window as any).goofi.query.graph().links);
}

/**
 * Pick `type` from the open add-node menu and place it with a tap on the canvas. Answers the
 * GHOST's rendered size, measured while it still exists: the finger carries the ghost by its middle,
 * so the corner the node commits at is half of that up and left of the tap — and a placed card is
 * NOT the same height as its ghost (it grows a viewer body), so its own box cannot stand in.
 */
async function pickAndPlace(
	page: Page,
	type: string,
	at: { x: number; y: number }
): Promise<{ width: number; height: number }> {
	await paletteItem(page, type).tap();
	const ghost = page.getByTestId('placement-ghost');
	await expect(ghost, `the ${type} ghost is following the finger`).toBeVisible();
	const size = (await ghost.boundingBox())!;
	await page.touchscreen.tap(at.x, at.y);
	await expect(ghost, 'the tap committed the placement').toHaveCount(0);
	return size;
}

test('412px portrait: add a node, connect it, open its parameters, change one, and save', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.

	// --- 1. ADD, through the coarse-pointer door -------------------------------------------
	const spot = await openAddMenuByPress(page);
	const ghost = await pickAndPlace(page, 'Oscillator', spot);
	await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);

	const osc = (await nodes(page))[0];
	const oscCard = page.locator(`.svelte-flow__node[data-id="${osc.uid}"]`);
	await expect(oscCard, 'the new node is on screen').toBeVisible();
	// …and it landed where the finger did. A touch device never reports a mouse position, so a
	// placement anchored on the last `mousemove` would drop every node at the viewport origin.
	// The finger holds the ghost by its MIDDLE (`touchPlacement.ts`'s `ghostOrigin`), so the corner
	// it commits at is half a GHOST up and left of the tap — `touch-placement.spec.ts` is where that
	// anchor is measured to the px; this is the end-to-end sanity check around it.
	const card = (await oscCard.boundingBox())!;
	expect(Math.abs(card.x - (spot.x - ghost.width / 2)), 'placed at the tap, horizontally').toBeLessThan(40);
	expect(Math.abs(card.y - (spot.y - ghost.height / 2)), 'placed at the tap, vertically').toBeLessThan(40);

	// --- 2. PARAMETERS — placing a node selects it, so its inspector is already up ------------
	const inspector = page.getByTestId('auto-side-panel');
	await expect(inspector, 'the placed node is inspected').toHaveClass(/open/);
	// This panel is portrait, so the pane is the bottom sheet: it spans the host's width and stops at
	// 60% of its height (D-I6), which means the canvas it leaves is ABOVE it rather than beside it.
	// Height is the safe read mid-slide too — a Y slide moves the box without resizing it.
	const box = (await inspector.boundingBox())!;
	const host = (await page.locator('.editor-panel').first().boundingBox())!;
	expect(box.height, 'and it leaves a strip of canvas above it').toBeLessThan(host.height);
	await expect(inspector.getByTestId('node-name')).toHaveText(osc.name);

	// --- 3. CHANGE ONE, through the rendered control -----------------------------------------
	const amp = inspector.getByTestId('param-field-amplitude').getByTestId('param-number');
	await amp.fill('0.37');
	await amp.press('Enter');
	await expect
		.poll(async () => (await nodeParams(page, osc.uid))?.oscillator?.amplitude?.value)
		.toBeCloseTo(0.37, 5);

	// --- 4. CONNECT — dismiss the inspector, then seed the next node from the output pill -----
	// The dismiss is load-bearing here, not decoration: at 412px the pane covers the canvas, so
	// without a way out the patch could not be grown past its first node (D-R9).
	await inspector.getByTestId('inspector-close').tap();
	await expect(inspector).toHaveCount(0);
	await oscCard.getByTestId('slot-output-pin').first().tap();
	await expect(page.getByTestId('add-menu-seed'), 'the menu opened seeded from that slot').toBeVisible();
	await pickAndPlace(page, 'Buffer', { x: spot.x, y: spot.y + 140 });
	await expect
		.poll(async () => (await links(page)).length, { message: 'the pick auto-wired the cable' })
		.toBe(1);
	expect((await links(page))[0].node_out).toBe(osc.uid);

	// --- 5. SAVE — from the bar, which is where Save lives at this width ----------------------
	// It used to spill, and this step used to reach it through the overflow menu. What moved is not
	// the overflow: the header carried an always-on "connected" chip whose 72px was exactly what
	// pushed Save out at 412px, and with the connection silent unless it BREAKS the bar keeps
	// Undo · Redo · Save · Load… here and gives up only the caret. Taking the door the user now has
	// is the point of the journey; the menu route is still pinned where it belongs —
	// `touch-reflow.spec.ts` asserts bar/menu parity at this very geometry, and
	// `topbar-overflow.spec.ts` opens the menu and reads its rows.
	const kept = await settledBar(page);
	expect(kept, 'at 412px the bar keeps Save itself; only the caret spills').toContain('topbar-save');
	await page.getByTestId('topbar-save').tap();

	const modal = page.getByTestId('fs-browser');
	await expect(modal, 'Save reached the file browser').toBeVisible();
	await expect(modal.getByTestId('fs-path-input')).not.toHaveValue('');
	const bar = modal.getByTestId('fs-path-input');
	await bar.fill(scratch);
	await bar.press('Enter');
	await expect(bar).toHaveValue(scratch);
	await modal.getByTestId('fs-filename').fill(patchName);
	await modal.getByTestId('fs-save').tap();
	await expect(modal, 'confirming Save closes the browser').toBeHidden();

	const patchFile = path.join(scratch, `${patchName}.gfi`);
	await expect.poll(() => fs.existsSync(patchFile), { message: 'the .gfi landed on disk' }).toBe(true);
	await expect
		.poll(() => page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges))
		.toBe(false);

	// --- hand the shared backend back unnamed and empty ---------------------------------------
	// One `new` does both halves; the `afterEach` below is the backstop for a failure before here.
	await resetPatch(page);
});
