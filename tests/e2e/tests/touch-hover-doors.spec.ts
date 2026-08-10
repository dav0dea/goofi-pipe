import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * R Task 4 — no interaction and no information may exist SOLELY behind hover (CLAUDE.md).
 *
 * Two shapes of door are proved here, both under real touch:
 *  - the app-wide `TitleTip` long-press layer, which is the ONE door built for the ~40 `title=`
 *    tooltips (rather than 40 bespoke coarse cues), and
 *  - the resting coarse form given to each control/label a fine pointer reveals on hover.
 *
 * Driven through CDP touch events like `touch-editor.spec.ts`: under `hasTouch` Playwright's mouse
 * API still dispatches MOUSE events, and `pointerType: 'mouse'` is exactly the input the long-press
 * layer stands down for — so a mouse-driven "press" would prove nothing.
 */

interface Pt {
	x: number;
	y: number;
}

async function touchSession(page: Page) {
	const cdp = await page.context().newCDPSession(page);
	const send = (type: string, touchPoints: Pt[]) =>
		cdp.send('Input.dispatchTouchEvent', { type, touchPoints } as never);
	return {
		down: (p: Pt) => send('touchStart', [p]),
		up: () => send('touchEnd', [])
	};
}

/** The centre of the first match, as a touch point. */
async function centreOf(page: Page, selector: string): Promise<Pt> {
	const box = (await page.locator(selector).first().boundingBox())!;
	expect(box, `${selector} is on screen`).toBeTruthy();
	return { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height / 2) };
}

/** The panel header's maximize button: on screen at this geometry, and it ACTS on tap — while
 * `maximizedPanelId` is provably not part of `WorkspaceState`, so nothing here persists.
 *
 * Named by its own test id, not as "the first `.hdr-btn`": the header's right end is a progressive
 * overflow now, so which control is first there is a function of the panel's width. Reaching for
 * an ordinal picked up Split Right instead, which both asked the wrong tooltip and — one test
 * later — really split the panel and left the arrangement behind. */
const MAX_BTN = '[data-testid="panel-maximize"]';

test('a long press reveals a title, and does not fire the control it asked about', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const touch = await touchSession(page);
	const tip = page.getByTestId('title-tip');

	const at = await centreOf(page, MAX_BTN);
	await touch.down(at);
	await expect(tip, 'the press surfaces the title the finger is on').toHaveText('Maximize');

	await touch.up();
	await expect(tip, 'and releasing leaves the answer standing').toHaveText('Maximize');
	// `toggleMaximize` would have flipped this to "Restore" had the click gone through. Asking what
	// a control does must never be the same as using it.
	await expect(
		page.locator(MAX_BTN).first(),
		'the press asked a question; it did not press the button'
	).toHaveAttribute('title', 'Maximize');

	// The next press anywhere dismisses — including one that resolves no title at all.
	await touch.down({ x: at.x, y: at.y + 200 });
	await touch.up();
	await expect(tip).toHaveCount(0);
});

test('a tap is not a press — the control acts and no tooltip appears', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const touch = await touchSession(page);
	const btn = page.locator(MAX_BTN).first();

	const at = await centreOf(page, MAX_BTN);
	await touch.down(at);
	await touch.up();
	await expect(btn, "a tap is still the control's own action").toHaveAttribute('title', 'Restore');
	await expect(page.getByTestId('title-tip')).toHaveCount(0);

	// Put it back — one backend serves the whole run.
	await touch.down(await centreOf(page, MAX_BTN));
	await touch.up();
	await expect(btn).toHaveAttribute('title', 'Maximize');
});

test('a node’s update rate is legible without hover', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// Buffer for the input slot (Oscillator has none); Oscillator for the rate (it is the one that
	// ticks unprompted, so it is the one that reports a rate).
	const buf = await addNode(page, 'Buffer', 'signal', [60, 60]);
	const osc = await addNode(page, 'Oscillator', 'inputs', [60, 260]);
	await waitForNode(page, buf);
	await waitForNode(page, osc);
	try {
		// The input NAME's coarse door used to be pinned here as "it rests open". It does not any
		// more: resting it open put a name tag on every input on the canvas at all times, so the door
		// is a proximity reveal scoped to a cable in flight — `touch-slot-name.spec.ts` owns both
		// halves of it, and this file keeps the rate, whose door really is a resting form.

		// The header rate sits at opacity .3 at rest and comes forward on hover — which on touch is
		// never. It appears with the node's first stats push.
		const rate = page.locator('.goofi-node .rate').first();
		await expect(rate).toBeVisible({ timeout: 15_000 });
		const op = await rate.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
		expect(op, 'the rate is readable without a hover to bring it forward').toBeGreaterThan(0.8);
	} finally {
		for (const u of [buf, osc]) {
			await page.evaluate((n) => (window as any).goofi.commands.removeNode(n), u);
			await waitForNoNode(page, u).catch(() => {});
		}
	}
});

test('a tap on a plot yields the per-sample readout hover used to own', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	try {
		const touch = await touchSession(page);
		const plot = page.locator(`.slot-viewer[data-node="${uid}"] .container`).first();
		await expect(plot).toBeVisible();
		const chip = page.getByTestId('cursor-chip');
		await expect(chip, 'nothing is pinned before the tap').toHaveCount(0);

		const box = (await plot.boundingBox())!;
		const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height / 2) };
		await touch.down(at);
		// Asserted WHILE THE FINGER IS DOWN, which is the whole discriminator: the browser's
		// compatibility `mousemove` only arrives after the release, so a `mousemove`-driven readout
		// cannot exist during the gesture — exactly when a scrub needs it.
		await expect(chip, 'the press pins a sample while the finger is still on it').toBeVisible();

		// …and it SURVIVES the release. A touch pointer is destroyed on lift, firing `pointerleave`
		// immediately; retracting there would make the only per-sample readout a flash.
		await touch.up();
		await expect(chip, 'lifting the finger keeps the readout').toBeVisible();
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});

test('the console per-row copy button is reachable without hover', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

	// Same content source as `console-rows.spec.ts`: a Python node whose process() needs a connected
	// ARRAY input raises every tick, and the graph store mirrors each raise into the console.
	const uid = await addNode(page, 'LempelZiv', 'python');
	try {
		await waitForNode(page, uid);
		const copy = page.getByTestId('console-copy').first();
		await expect(copy).toBeVisible();
		await expect(copy, 'the copy control rests visible where there is no hover').toHaveCSS(
			'opacity',
			'1'
		);
		const events = await copy.evaluate((el) => getComputedStyle(el).pointerEvents);
		expect(events, 'and it is tappable, not merely painted').not.toBe('none');

		// …and it is a GHOST, like every other chrome-density icon button in the app. Resting it
		// open was the right call; inheriting `.ui-icon-btn`'s filled + outlined base paint with it
		// was not. On a fine pointer that is invisible (16px at `opacity: 0`); under coarse it puts
		// a painted 44×44 box on every console row — the highest repetition rate in the app, against
		// app.css's own rule that a surface step is what carries separation.
		const paint = await copy.evaluate((el) => {
			const cs = getComputedStyle(el);
			return { border: cs.borderTopColor, bg: cs.backgroundColor };
		});
		const invisible = /rgba\(0, 0, 0, 0\)|transparent/;
		expect(paint.border, 'the resting copy button draws no border').toMatch(invisible);
		expect(paint.bg, 'nor a filled surface').toMatch(invisible);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		// The panel type is a command, and the assertion above only passes once the manager's own
		// delta drew it back — so the running patch already holds the restore.
	}
});

test('the inspector resize seam is visible without hover', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	try {
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		const handle = page.getByTestId('panel-resize-handle');
		await expect(handle).toBeVisible();
		const painted = await handle.evaluate(
			(el) => getComputedStyle(el, '::after').backgroundColor
		);
		expect(painted, 'the seam paints a line at rest, not only on hover').not.toBe(
			'rgba(0, 0, 0, 0)'
		);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});
