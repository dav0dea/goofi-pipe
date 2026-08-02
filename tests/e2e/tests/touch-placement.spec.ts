import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { nodes } from '../lib/goofi';
import { openGhost } from '../lib/placement';
import { touchSession, type TouchPoint } from '../lib/touch';

/**
 * Placing a new node WITH A FINGER — the gesture, and the two halves of it that are easy to confuse.
 *
 * Adding a node spawns a ghost that follows the pointer until it is put down. On a phone that was
 * broken in both directions at once: `PlacementPreview` listened for `mousemove`/`mousedown`/`click`
 * and nothing else, so no finger ever moved the ghost, and the drag it did not consume reached
 * SvelteFlow's panner instead — the canvas slid out from under a ghost frozen where the palette row
 * had been, and the tap that followed committed the node there.
 *
 * SO BOTH HALVES ARE ASSERTED, SEPARATELY AND ON PURPOSE. The ghost renders inside a
 * `ViewportPortal`, i.e. in FLOW coordinates, so "the ghost moved" and "the canvas panned" produce
 * the SAME screen-space delta and a spec that measures only the ghost passes on the old behaviour.
 * What pins it is the pair: the ghost's screen box tracks the finger, AND `.svelte-flow__viewport`'s
 * transform is byte-identical before and after.
 *
 * MODALITY, NOT ORIENTATION — the rule `panels/paneDrag.ts` and `touch-modality.spec.ts` are built
 * on: orientation decides an anchored axis, input modality decides a gesture. This gesture is
 * modality-gated (`pointerType === 'touch'`, asked per event so a hybrid device stays right on both
 * of its inputs), which means it must answer identically in either anchor — so `playwright.config.ts`
 * runs these same bytes in `touch` AND `touch-landscape`. A re-coupling to orientation fails one
 * project by name instead of going unnoticed.
 */

test.beforeEach(async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await clearGraph(page);
});

/** Hand the shared backend back empty even on failure — a leftover card changes where the NEXT
 *  spec's `emptySpot` may press, and adds a snap target that moves this one's arithmetic. */
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

/** Two points inside the flow pane, far enough apart to be a drag in EITHER anchor — read off the
 *  pane itself rather than hardcoded, since portrait and landscape share these bytes. */
async function dragPoints(page: Page): Promise<{ from: TouchPoint; to: TouchPoint }> {
	const p = (await page.locator('.svelte-flow__pane').first().boundingBox())!;
	const at = (fx: number, fy: number): TouchPoint => ({
		x: Math.round(p.x + p.width * fx),
		y: Math.round(p.y + p.height * fy)
	});
	return { from: at(0.25, 0.3), to: at(0.6, 0.6) };
}

/** The pan/zoom matrix every flow-space thing is drawn through. */
const viewport = (page: Page): Locator => page.locator('.svelte-flow__viewport');

async function boxAt(ghost: Locator): Promise<{ x: number; y: number }> {
	const b = (await ghost.boundingBox())!;
	return { x: b.x, y: b.y };
}

/**
 * The ghost's top-left IS the point the finger is at: it is drawn at the flow position
 * `screenToFlowPosition` mapped that client point to, so the round trip lands back on it.
 *
 * The graph is emptied first, so there are no snap targets and the position is the finger's
 * unmodified — the few px of slack are the commit's rounding to whole flow units, taken back
 * through a 0.85 zoom.
 */
const TOL = 4;
function expectUnderTheFinger(box: { x: number; y: number }, at: TouchPoint, what: string): void {
	expect(Math.abs(box.x - at.x), `${what}: horizontally (at ${box.x}, wanted ${at.x})`).toBeLessThanOrEqual(TOL);
	expect(Math.abs(box.y - at.y), `${what}: vertically (at ${box.y}, wanted ${at.y})`).toBeLessThanOrEqual(TOL);
}

test('a touch drag moves the GHOST and leaves the canvas exactly where it was', async ({ page }) => {
	const ghost = await openGhost(page, 'Oscillator');
	const { from, to } = await dragPoints(page);
	const touch = await touchSession(page);

	// The matrix BEFORE anything is touched. Read as a string and compared as one: a pan of a single
	// px changes it, and nothing else on this page writes it.
	const before = await viewport(page).evaluate((el) => getComputedStyle(el).transform);

	await touch.down(from);
	expectUnderTheFinger(await boxAt(ghost), from, 'the press put the ghost under the finger');

	for (let i = 1; i <= 5; i++) {
		await touch.moveTo({
			x: Math.round(from.x + ((to.x - from.x) * i) / 5),
			y: Math.round(from.y + ((to.y - from.y) * i) / 5)
		});
	}
	expectUnderTheFinger(await boxAt(ghost), to, 'and it tracked the finger across the canvas');

	// THE HALF THAT REGRESSES SILENTLY. Both a moving ghost and a panning canvas move the ghost's
	// screen box by the drag delta, so the assertion above cannot tell them apart — this one can.
	await expect(
		viewport(page),
		'the drag moved the ghost, NOT the canvas — SvelteFlow never saw it'
	).toHaveCSS('transform', before);

	// Chromium reads back-to-back synthetic moves as a FLING, which then eats the next tap anywhere
	// on the page. Letting the finger come to rest before it lifts is what keeps this spec from
	// poisoning the one after it.
	await page.waitForTimeout(150);
	await touch.up();

	await expect(ghost, 'lifting the finger committed the placement').toHaveCount(0);
	await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);
	const uid = (await nodes(page))[0].uid;
	const card = page.locator(`.svelte-flow__node[data-id="${uid}"]`);
	expectUnderTheFinger(await boxAt(card), to, 'and it landed where the finger was released');
	await expect(viewport(page), 'and the canvas still has not moved').toHaveCSS('transform', before);
});

test('a plain tap places the node at the tap, on the same path as the drag', async ({ page }) => {
	const ghost = await openGhost(page, 'Oscillator');
	const { to } = await dragPoints(page);

	// No move between down and up at all: a tap is a drag of zero length, which is the whole reason
	// there is no tap-vs-drag threshold for it to fall on the wrong side of.
	await page.touchscreen.tap(to.x, to.y);

	await expect(ghost, 'the tap committed the placement').toHaveCount(0);
	await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);
	const uid = (await nodes(page))[0].uid;
	expectUnderTheFinger(
		await boxAt(page.locator(`.svelte-flow__node[data-id="${uid}"]`)),
		to,
		'placed at the tap'
	);
});

/* The comparison is the SAME NODE before and after it is put down, which is the only way to state
   "this one is not placed yet" as a measurement rather than as a constant. A second card measured
   beside the ghost would say the same thing and would also have to be manoeuvred out of the way of
   the next `emptySpot`. */
test('the pending ghost reads as a placeholder, and the node it becomes does not', async ({
	page
}) => {
	const opacity = (loc: Locator): Promise<number> =>
		loc.evaluate((el) => parseFloat(getComputedStyle(el).opacity));

	const ghost = await openGhost(page, 'Oscillator');
	const pending = await opacity(ghost);

	const { to } = await dragPoints(page);
	await page.touchscreen.tap(to.x, to.y);
	await expect(ghost).toHaveCount(0);
	await expect.poll(async () => (await nodes(page)).length).toBe(1);
	const card = page.locator(`.svelte-flow__node[data-id="${(await nodes(page))[0].uid}"]`);

	expect(await opacity(card), 'a placed node is drawn solid').toBe(1);
	expect(
		pending,
		'a coarse pointer has no cursor to say "this is stuck to me", so the ghost must say it itself'
	).toBeLessThanOrEqual(0.7);
});
