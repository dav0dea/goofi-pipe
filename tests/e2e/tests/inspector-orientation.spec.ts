import { test, expect, type Page } from '@playwright/test';
import { closeSplit, splitRight, waitForApp } from '../lib/app';
import { settledBox } from '../lib/geometry';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * The orientation-aware inspector, on the FINE pointer — because the anchor rule contains no device
 * class at all (D-I2). It asks its host panel whether it is taller than it is wide, through
 * `@container (orientation: portrait)`, and a desktop panel can answer either way.
 *
 * That is the point of this file being a `default`-project spec: everything here runs on a mouse,
 * at a landscape desktop window, and one of the cases still gets the bottom sheet. `touch-sheet` and
 * `touch-reflow` cover the phone geometries.
 *
 * Sizes are measured against the HOST, never a literal — except where the literal IS the claim
 * (§4.3: the desktop resting width does not move off 420px), and there the viewport is set to a
 * width where `min(30%, 30rem)` provably resolves to the rem half.
 */

const pane = (page: Page) => page.getByTestId('auto-side-panel').first();
const editor = (page: Page) => page.locator('.editor-panel').first();

/** The root font size the responsive `clamp()` settled on — the px `30rem` is measured in. */
const rem = (page: Page): Promise<number> =>
	page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).fontSize));

/** Boot, add an Oscillator, select it so the inspector slides in, and return its uid. The pane
 *  slides over `--dur-slow` and every number here is a POSITION, so the app's own reduced-motion
 *  rule collapses the transition first — a mid-slide read describes a frame, not the layout. */
async function addAndSelect(page: Page): Promise<string> {
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	// `.open`, not merely visible: the pane is MOUNTED at every moment and parked off-edge until a
	// selection lands, so `toBeVisible` resolves on a pane that is still off-screen.
	await expect(pane(page), 'a single selection opens the inspector').toHaveClass(/open/);
	return uid;
}

async function drop(page: Page, uid: string): Promise<void> {
	await page.evaluate(() => (window as any).goofi.commands.clearSelection());
	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid).catch(() => {});
}

test('the desktop resting width does not move off 420px (§4.3)', async ({ page }) => {
	// 1600 wide, so `min(30%, 30rem)` resolves to the REM half: 30% of a ~1592px editor is 478px,
	// and the root clamp saturates at 14px anywhere near this size, so 30rem is exactly 420 — the
	// width the pane has always rested at. Below ~1400px the percent half binds instead, which is
	// the next case.
	await page.setViewportSize({ width: 1600, height: 900 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	expect(await rem(page), 'the root clamp is saturated at this size').toBeCloseTo(14, 5);
	const uid = await addAndSelect(page);
	try {
		const p = await settledBox(pane(page));
		expect(p.width, 'the resting pane is 30rem = 420px').toBeCloseTo(420, 0);
		const host = (await editor(page).boundingBox())!;
		expect(host.width * 0.3, 'and the percent half is the looser of the two here').toBeGreaterThan(
			420
		);
	} finally {
		await drop(page, uid);
	}
});

test('a narrower desktop editor caps the pane at 30% of it, not at 720px', async ({ page }) => {
	// The default 1280 viewport: 30% of the editor is ~382px, under the 410px `30rem` resolves to at
	// this root size, so the PERCENT half binds. Before D-I6 the pane sat at a flat 420px here and
	// its only host clamp was `100% - --hit`, which reserved 28px of canvas out of 1272.
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const host = (await editor(page).boundingBox())!;
		const cap = Math.min(0.3 * host.width, 30 * (await rem(page)));
		expect(cap, 'the percent half is the tighter of the two here').toBeLessThan(420);
		const p = await settledBox(pane(page));
		expect(Math.abs(p.width - cap), `pane ${p.width} vs cap ${cap}`).toBeLessThan(1);
	} finally {
		await drop(page, uid);
	}
});

/**
 * The consequence the spec flagged for the user rather than choosing silently (§6), pinned so it
 * reads as deliberate.
 *
 * The WINDOW is landscape here — `matchMedia` says so — and the pane is a bottom sheet anyway,
 * because the panel it lives in is a narrow tall column. That is the whole argument for D-I2 over
 * `@media (orientation)`: this docked editor has exactly the phone's problem (a right-hand pane
 * would eat the canvas), and a viewport query would answer "landscape" and leave it broken. It is
 * also the one line that would be edited to take it back — see `InspectorOverlay.svelte`'s
 * `@media all` prelude.
 */
test('a narrow, tall DOCKED editor gets the sheet, in a landscape window (spec §6)', async ({
	page
}) => {
	await page.setViewportSize({ width: 1280, height: 900 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	expect(
		await page.evaluate(() => matchMedia('(orientation: landscape)').matches),
		'the WINDOW is landscape — the sheet below is the panel’s shape, not the screen’s'
	).toBe(true);
	await splitRight(page);
	let uid = '';
	try {
		// Both halves are node editors and each mounts its own inspector, so everything below is
		// scoped to the LEFT one — and its selection is addressed by panel id rather than through the
		// active editor, which the split just moved to the new panel.
		const left = page.locator('.panel').first();
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		await page.evaluate(
			([u, id]) => (window as any).goofi.commands.select([u], id),
			[uid, panelId] as const
		);
		const sheet = left.getByTestId('auto-side-panel');
		await expect(sheet, 'the left editor inspects its selection').toHaveClass(/open/);

		const host = (await left.locator('.editor-panel').boundingBox())!;
		expect(host.height, 'the split left the editor taller than it is wide').toBeGreaterThan(
			host.width
		);

		const p = await settledBox(sheet);
		expect(p.width, 'the sheet spans its host').toBeCloseTo(host.width, 0);
		expect(
			host.y + host.height - (p.y + p.height),
			'and is flush with the bottom it slid from'
		).toBeLessThan(2);
		expect(p.height, 'at the 60% D-I6 allows').toBeCloseTo(host.height * 0.6, 0);
	} finally {
		if (uid) await drop(page, uid);
		await closeSplit(page);
	}
});

/**
 * D-I3/D-I4: the edge drag is not a touch affordance. It is THE resize, identical on both inputs,
 * and this is the mouse half — driven with `page.mouse`, whose `pointerType` is `mouse`, i.e. the
 * input every coarse door in the app is closed to.
 */
test('an edge drag resizes the pane with a MOUSE, and the size outlives the reload', async ({
	page
}) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const before = await settledBox(pane(page));
		const grip = (await page.getByTestId('panel-resize-handle').boundingBox())!;
		const y = grip.y + grip.height / 2;
		// Rightward, i.e. INTO the pane, which shrinks it — the same direction of travel that shrinks
		// the sheet when it is pushed down. Away from the pane the ceiling would bind and the
		// measurement would be of `max-width`, not of the drag.
		await page.mouse.move(grip.x + grip.width / 2, y);
		await page.mouse.down();
		await page.mouse.move(grip.x + grip.width / 2 + 60, y, { steps: 8 });
		await page.mouse.up();

		const after = await settledBox(pane(page));
		expect(after.width, 'the pane shrank by exactly the drag').toBeCloseTo(before.width - 60, 0);
		expect(
			await page.evaluate(() => localStorage.getItem('goofi.panelWidth')),
			'and the RENDERED size is what was stored, so a reload agrees with the screen'
		).toBe(String(Math.round(after.width)));

		await page.reload();
		await waitForApp(page);
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		await expect(pane(page)).toHaveClass(/open/);
		const restored = await settledBox(pane(page));
		expect(restored.width, 'and it comes back at that width').toBeCloseTo(after.width, 0);
	} finally {
		await drop(page, uid);
	}
});

/**
 * D-I9. The desktop grip is a transparent line until it is hovered, which is the whole of its
 * discoverability — and CLAUDE.md forbids an affordance that exists solely behind `:hover`. A sheet
 * with no visible grabber also simply does not read as draggable. On a FINE pointer here, so what
 * is proved is the portrait rule and not the coarse one `touch-hover-doors.spec.ts` already covers.
 */
test('the sheet rests with a visible grabber, not an invisible seam', async ({ page }) => {
	await page.setViewportSize({ width: 600, height: 1000 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		// The pointer is parked off the pane — no hover anywhere near the grip.
		await page.mouse.move(5, 5);
		const pill = await page.getByTestId('panel-resize-handle').evaluate((el) => {
			const cs = getComputedStyle(el, '::after');
			return { bg: cs.backgroundColor, width: parseFloat(cs.width), height: parseFloat(cs.height) };
		});
		expect(pill.bg, 'the grabber paints at rest').not.toBe('rgba(0, 0, 0, 0)');
		expect(pill.width, 'and it is a pill across the sheet, not a hairline down its side').
			toBeGreaterThan(pill.height * 4);
	} finally {
		await drop(page, uid);
	}
});

test('the ✕ dismisses the pane in either anchor', async ({ page }) => {
	// `inspector-dismiss.spec.ts` proves the ✕ on the right-hand pane; this is the same door on the
	// SHEET, and D-I4 is why it must exist there too — the swipe is an extra, never the only way out.
	await page.setViewportSize({ width: 600, height: 1000 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		await expect
			.poll(() =>
				pane(page).evaluate((el) => getComputedStyle(el).getPropertyValue('--pane-axis').trim())
			)
			.toBe('y');
		await pane(page).getByTestId('inspector-close').click();
		await expect(pane(page), 'the sheet closes').toHaveCount(0);
	} finally {
		await drop(page, uid);
	}
});
