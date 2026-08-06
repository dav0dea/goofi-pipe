import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { nodes } from '../lib/goofi';
import { emptySpot, touchSession, type TouchPoint } from '../lib/touch';

/**
 * DOUBLE-TAP-AND-DRAG TO ZOOM — the one-handed gesture every map app has, ADDED beside pinch.
 *
 * Two fingers were the only way to zoom this canvas, which on a phone means putting down whatever
 * you are holding. The map gesture is the standard answer: tap, then tap again and keep the finger
 * down, and dragging it moves the zoom — with the point you tapped held still under it, so you are
 * magnifying the thing you pointed at rather than the middle of the screen.
 *
 * WHAT THIS FILE HAS TO PIN, and why each half regresses on its own:
 *
 *  · THE ZOOM ITSELF. `.svelte-flow__viewport`'s matrix is the one place the answer lives, and its
 *    SCALE is the only part of it a pan cannot move. Read before and after.
 *
 *  · THE ANCHOR. A zoom about the screen centre also changes the scale, so the scale assertion
 *    alone cannot tell an anchored zoom from an unanchored one. What can: the flow point drawn
 *    under the tapped screen point, computed from the matrix before and after — it is the same
 *    point if and only if the zoom was taken about it.
 *
 *  · THAT ONE FINGER STILL PANS. A recognizer that swallows the FIRST touch takes panning with it,
 *    and nothing else in this suite would notice: a canvas that no longer pans still looks like a
 *    canvas. So the guard is a plain tap-drag, asserting the matrix TRANSLATED and its scale did
 *    not move — the mirror image of the assertions above.
 *
 * Pinch is untouched (`zoomOnPinch` is still SvelteFlow's own), and this gesture is deliberately
 * additive: the seam it uses is `zoomOnDoubleClick={false}`, i.e. a double tap that was already
 * inert.
 */

test.beforeEach(async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await clearGraph(page);
});

/** Hand the shared backend back empty even on failure — a leftover card is a tap target the next
 *  spec's `emptySpot` has to work around. */
test.afterEach(async ({ page }) => {
	await clearGraph(page).catch(() => {});
});

async function clearGraph(page: Page): Promise<void> {
	await page.evaluate(() => {
		const g = (window as any).goofi;
		const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
		if (uids.length) return g.commands.removeNodes(uids);
	});
	await expect.poll(async () => (await nodes(page)).length).toBe(0);
}

/** The pan/zoom matrix every flow-space thing is drawn through. */
const viewport = (page: Page): Locator => page.locator('.svelte-flow__viewport');
/** Its untransformed parent — so the pane's own top-left IS the matrix's origin. */
const pane = (page: Page): Locator => page.locator('.svelte-flow__pane').first();

interface Matrix {
	/** The zoom. */
	k: number;
	x: number;
	y: number;
}

async function matrixOf(page: Page): Promise<Matrix> {
	return viewport(page).evaluate((el) => {
		const m = new DOMMatrix(getComputedStyle(el).transform);
		return { k: m.a, x: m.e, y: m.f };
	});
}

const scaleOf = async (page: Page): Promise<number> => (await matrixOf(page)).k;

/** The FLOW point drawn under a screen point — the matrix, inverted. */
function flowUnder(m: Matrix, origin: TouchPoint, at: TouchPoint): TouchPoint {
	return { x: (at.x - origin.x - m.x) / m.k, y: (at.y - origin.y - m.y) / m.k };
}

/** Feed a finger from `at` to `at + (0, dy)` in steps, then let it come to rest. */
async function dragBy(
	page: Page,
	touch: { moveTo(p: TouchPoint): Promise<unknown> },
	at: TouchPoint,
	dy: number
): Promise<void> {
	for (let i = 1; i <= 5; i++) {
		await touch.moveTo({ x: at.x, y: at.y + Math.round((dy * i) / 5) });
	}
	// Chromium reads back-to-back synthetic moves as a FLING, which then eats the next tap anywhere
	// on the page. Coming to rest before the lift is what keeps this spec from poisoning the next.
	await page.waitForTimeout(150);
}

test('a double tap and a drag zooms the canvas, about the point that was tapped', async ({
	page
}) => {
	const box = (await pane(page).boundingBox())!;
	const origin = { x: box.x, y: box.y };
	// Low in the pane, so the upward drag has room without leaving the screen. Asserted bare rather
	// than assumed: a tap that lands on a node card is a node drag, and would green nothing.
	const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height * 0.75) };
	await expect
		.poll(() =>
			page.evaluate(
				(p) => document.elementFromPoint(p.x, p.y)?.classList.contains('svelte-flow__pane'),
				at
			))
		.toBe(true);

	const before = await matrixOf(page);
	const held = flowUnder(before, origin, at);

	const touch = await touchSession(page);
	await touch.down(at);
	await touch.up();
	// The second tap, then HOLD, then drag — the gesture as a hand actually performs it, and the
	// hold is not incidental: 600 ms is past the 500 ms long press, so this is the arrangement in
	// which the add-node menu would open under the finger if the gesture did not disarm it.
	await touch.down(at);
	await page.waitForTimeout(600);
	await dragBy(page, touch, at, -Math.min(200, Math.round(box.height * 0.4)));
	await touch.up();

	await expect
		.poll(() => scaleOf(page), { message: 'dragging up zoomed in' })
		.toBeGreaterThan(before.k * 1.5);

	// The two things a finger held on this canvas ALSO means, neither of which may fire here: the
	// long press that is the coarse door onto the add-node menu (armed on the very `pointerdown`
	// that starts this gesture), and the compat `dblclick` a double tap would otherwise replay onto
	// the pane, which opens the same menu by the mouse route.
	await expect(page.getByTestId('add-node-menu-anchor'), 'zooming is not asking for a node').toHaveCount(0);

	const now = flowUnder(await matrixOf(page), origin, at);
	expect(Math.abs(now.x - held.x), 'the tapped point stayed under the finger, horizontally').toBeLessThan(4);
	expect(Math.abs(now.y - held.y), 'the tapped point stayed under the finger, vertically').toBeLessThan(4);
});

test('and dragging the other way zooms back out', async ({ page }) => {
	const box = (await pane(page).boundingBox())!;
	const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height * 0.3) };
	const before = await scaleOf(page);

	const touch = await touchSession(page);
	await touch.down(at);
	await touch.up();
	await touch.down(at);
	await dragBy(page, touch, at, Math.min(200, Math.round(box.height * 0.4)));
	await touch.up();

	await expect
		.poll(() => scaleOf(page), { message: 'dragging down zoomed out' })
		.toBeLessThan(before * 0.7);
});

test('a single tap and drag still PANS, and does not zoom', async ({ page }) => {
	// The regression this gesture could plausibly cause, and the one nothing else would catch.
	const at = await emptySpot(page);
	const before = await matrixOf(page);

	const touch = await touchSession(page);
	await touch.down(at);
	await dragBy(page, touch, at, 120);
	await touch.up();

	await expect
		.poll(() => matrixOf(page).then((m) => m.y - before.y), { message: 'one finger panned' })
		.toBeGreaterThan(60);
	expect(await scaleOf(page), 'and panning is not zooming').toBeCloseTo(before.k, 5);
});
