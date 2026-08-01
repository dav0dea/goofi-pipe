import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { settledBox } from '../lib/geometry';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { emptySpot, setKeyboardInset, touchSession } from '../lib/touch';

/**
 * The inspector as a bottom sheet — the `touch` (Pixel 7 portrait) project, where the host editor
 * panel is decisively taller than it is wide and `@container (orientation: portrait)` fires.
 *
 * This is not a touch-only MODE (the governing principle of the spec): nothing asserted here is
 * gated on the pointer, and `inspector-orientation.spec.ts` proves the very same sheet on a mouse at
 * a landscape desktop window. What this file adds is the geometry only a phone has — 412×915, where
 * a right-anchored pane left a 44px strip of canvas — plus the soft keyboard, which no desktop
 * project can fake honestly.
 *
 * The pane slides over `--dur-slow`, so every position here is read past the app's own
 * reduced-motion collapse and through a settling read; a one-shot measurement mid-slide describes a
 * frame of the animation, not the layout.
 */

const pane = (page: Page) => page.getByTestId('auto-side-panel');
const editor = (page: Page) => page.locator('.editor-panel').first();

/** Which axis the container query put the pane on — the same custom property the drag handler
 *  reads, so a test cannot agree with the layout while disagreeing with the gesture. */
const paneAxis = (page: Page): Promise<string> =>
	pane(page).evaluate((el) => getComputedStyle(el).getPropertyValue('--pane-axis').trim());

/** The soft-keyboard overlap the app published, in px. */
const kbInset = (page: Page): Promise<number> =>
	page.evaluate(
		() => parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--kb-inset')) || 0
	);

async function addAndSelect(page: Page): Promise<string> {
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(pane(page), 'a single selection opens the inspector').toHaveClass(/open/);
	return uid;
}

async function drop(page: Page, uid: string): Promise<void> {
	await page.evaluate(() => (window as any).goofi.commands.clearSelection());
	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid).catch(() => {});
}

test('the pane rises from the bottom and stops at 60% of its host (D-I6)', async ({ page }) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		// The axis is a CSS fact, and the one the drag handler reads back — so read it the same way
		// rather than inferring it from a box that has already been laid out.
		await expect
			.poll(() => paneAxis(page), { message: 'the sheet slid up, not in from the right' })
			.toBe('y');
		const host = (await editor(page).boundingBox())!;
		const p = await settledBox(pane(page));
		expect(p.width, 'it spans the host').toBeCloseTo(host.width, 0);
		expect(p.x, 'from its left edge').toBeCloseTo(host.x, 0);
		expect(host.y + host.height - (p.y + p.height), 'flush with the bottom').toBeLessThan(2);
		expect(p.height, 'and 60% of it tall').toBeCloseTo(host.height * 0.6, 0);
	} finally {
		await drop(page, uid);
	}
});

test('the canvas above the sheet still takes a tap that changes the selection', async ({ page }) => {
	// Success criterion §4.1, and the whole reason the pane is capped at all: before R the pane's only
	// host clamp reserved one `--hit`, which on a 412px phone is a 44px strip — and a strip is the
	// only way out, since deselecting means tapping canvas the pane covers.
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const p = await settledBox(pane(page));
		// `emptySpot` only resolves where the flow pane is really the TOPMOST element, so a point it
		// returns is by construction neither under the sheet nor under its grip band — and the 40% of
		// host it has to find one in is exactly the canvas the 60% cap hands back. It also clears
		// every node card by a tap target, because Chromium's touch adjustment would otherwise snap
		// a tap taken beside a card ONTO it and re-select the node this tap means to drop.
		const spot = await emptySpot(page);
		expect(spot.y, 'the live canvas it found is above the sheet').toBeLessThan(p.y);
		await page.touchscreen.tap(spot.x, spot.y);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length), {
				message: 'the tap reached the canvas and changed the selection'
			})
			.toBe(0);
	} finally {
		await drop(page, uid);
	}
});

/**
 * The other half of D-I4's "edge drag is for BOTH": the same grip, the same module, the same
 * persistence idiom — a finger instead of a mouse, and the Y axis instead of X. Nothing about this
 * gesture is gated on the pointer; what touch adds is the swipe layered on top of it.
 *
 * Real CDP touch, not `page.mouse`: under `hasTouch` Playwright's mouse API still reports
 * `pointerType: 'mouse'`, which would prove the desktop path a second time.
 */
test('an edge drag resizes the sheet by TOUCH — the same gesture, the other axis (D-I3)', async ({
	page
}) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const before = await settledBox(pane(page));
		const grip = (await page.getByTestId('panel-resize-handle').boundingBox())!;
		const x = Math.round(grip.x + grip.width / 2);
		const y0 = Math.round(grip.y + grip.height / 2);
		const touch = await touchSession(page);
		// DOWN the screen, i.e. into the sheet, which shrinks it. Upward the 60% ceiling would bind
		// and what came back would be a measurement of `max-height`, not of the drag.
		await touch.down({ x, y: y0 });
		for (let i = 1; i <= 5; i++) await touch.moveTo({ x, y: y0 + i * 20 });
		await touch.up();

		const after = await settledBox(pane(page));
		expect(after.height, 'the sheet shrank by exactly the drag').toBeCloseTo(before.height - 100, 0);
		expect(after.width, 'and stayed the full width of its host').toBeCloseTo(before.width, 0);
		expect(
			await page.evaluate(() => localStorage.getItem('goofi.panelHeight')),
			'persisted under its OWN key, the same idiom as the width (D-I3)'
		).toBe(String(Math.round(after.height)));
		expect(
			await page.evaluate(() => localStorage.getItem('goofi.panelWidth')),
			'and the other axis was not touched'
		).toBeNull();
	} finally {
		await drop(page, uid);
	}
});

/**
 * The additive half of D-I4, and the only thing in this pane that input modality gates: the SAME
 * drag, carried far enough past the floor, closes the sheet instead of resizing it.
 *
 * `inspector-orientation.spec.ts` runs the mirror of this under a mouse and proves the pane stays
 * open — one gesture, one extra meaning, and only for a finger.
 */
test('a swipe carried past the floor dismisses the sheet (D-I4)', async ({ page }) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const before = await settledBox(pane(page));
		const grip = (await page.getByTestId('panel-resize-handle').boundingBox())!;
		const x = Math.round(grip.x + grip.width / 2);
		const y0 = Math.round(grip.y + grip.height / 2);
		// Far enough that the size the swipe ASKS for is a full overshoot under the floor — which is
		// the only thing that distinguishes it from a resize that merely bottomed out, since the pane
		// clamps and the two look identical on screen.
		const throwBy = before.height + 100;
		const touch = await touchSession(page);
		await touch.down({ x, y: y0 });
		for (let i = 1; i <= 8; i++) await touch.moveTo({ x, y: Math.round(y0 + (throwBy * i) / 8) });
		await touch.up();

		// Dismissing turns the inspector OFF, which unmounts the pane — the canvas is genuinely
		// handed back, exactly as the ✕ hands it back.
		await expect(pane(page), 'the swipe threw the sheet away').toHaveCount(0);
		expect(
			await page.evaluate(() => localStorage.getItem('goofi.panelHeight')),
			'and a dismiss is not a resize: nothing was persisted, so it comes back its old size'
		).toBeNull();
	} finally {
		// The dismiss flipped this editor's inspector off, which is per-PAGE state (`selection`'s
		// `inspectorOn`, never the layout blob) — so the next spec's fresh page has it on again and
		// there is nothing to hand back but the node.
		await drop(page, uid);
	}
});

test('the soft keyboard lifts the sheet off the bottom (D-I7)', async ({ page }) => {
	// `--kb-inset` is the one thing R kept of the device seam, precisely because CSS cannot see the
	// soft keyboard — and a text field inside a bottom sheet is the case that needed it. This is its
	// fourth consumer.
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addAndSelect(page);
	try {
		const host = (await editor(page).boundingBox())!;
		const down = await settledBox(pane(page));
		expect(host.y + host.height - (down.y + down.height), 'resting on the bottom').toBeLessThan(2);

		const KB = 260;
		await setKeyboardInset(page, KB);
		await expect.poll(() => kbInset(page), { message: 'the app measured the keyboard' }).toBe(KB);
		const up = await settledBox(pane(page));
		expect(
			host.y + host.height - (up.y + up.height),
			'the sheet sits a keyboard above the bottom'
		).toBeCloseTo(KB, 0);
		// …and it did not merely grow upward past its cap to do it.
		expect(up.height, 'still 60% tall').toBeCloseTo(down.height, 0);
	} finally {
		await setKeyboardInset(page, 0);
		await drop(page, uid);
	}
});
