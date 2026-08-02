import { expect, type Locator, type Page } from '@playwright/test';
import { settledBox } from './geometry';
import { addNode, waitForNode, waitForNoNode } from './goofi';
import { touchSession } from './touch';

/**
 * The inspector pane, and THE ONE RULE it is built on:
 *
 *   ORIENTATION decides only the AXIS the pane is anchored on — portrait a bottom sheet,
 *   landscape/desktop the right-hand edge. INPUT MODALITY decides the GESTURE and its AFFORDANCE —
 *   the edge drag is for a mouse AND a finger, the swipe is the finger's extra, and the resting
 *   grabber is what a pointer with no hover needs to see. The two are INDEPENDENT: the modality
 *   logic is identical in portrait and in landscape.
 *
 * Which is why every assertion here is written ONCE and normalised to whichever axis the pane came
 * back anchored on, then invoked from both orientations — `touch` (Pixel 7 portrait, the sheet) and
 * `touch-landscape` (the right-hand pane) run the same bytes, not two copies that can drift apart.
 * A helper shaped this way cannot state an orientation-coupled expectation, so a change that
 * re-couples the two makes exactly one of those projects disagree, on the next run.
 *
 * `inspector-orientation.spec.ts` is the fine-pointer half and shares the same readers, so the
 * mouse and the finger are measured with one ruler too.
 */

export type PaneAxis = 'x' | 'y';

export const pane = (page: Page): Locator => page.getByTestId('auto-side-panel').first();
export const grip = (page: Page): Locator => page.getByTestId('panel-resize-handle').first();
export const editorHost = (page: Page): Locator => page.locator('.editor-panel').first();

/**
 * What the axis selects, mirroring `frontend/src/lib/panels/paneDrag.ts`'s `PANE_AXES`. The e2e
 * package cannot import the app's source, so this is the ONE place those numbers are restated —
 * rather than the several spec sites that each carried their own copy of `260`.
 */
export const PANE_AXES: Record<PaneAxis, { min: number; key: string }> = {
	x: { min: 260, key: 'goofi.panelWidth' },
	y: { min: 160, key: 'goofi.panelHeight' }
};

/** `paneDrag.ts`'s `DISMISS_OVERSHOOT_PX`: how far past the floor a swipe must be carried before it
 *  is a dismiss rather than a resize that merely bottomed out. */
export const DISMISS_OVERSHOOT_PX = 44;

/** The root font size the responsive `clamp()` settled on — the px `2.5rem` and `30rem` are
 *  measured in. */
export const rootRem = (page: Page): Promise<number> =>
	page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).fontSize));

/** Which axis the container query anchored the pane on — read off `--pane-axis`, the same property
 *  the drag handler reads back, so a test cannot agree with the layout while disagreeing with the
 *  gesture. */
export async function paneAxis(page: Page): Promise<PaneAxis> {
	const v = await pane(page).evaluate((el) =>
		getComputedStyle(el).getPropertyValue('--pane-axis').trim()
	);
	expect(v, 'the pane publishes the anchor it settled on').toMatch(/^[xy]$/);
	return v as PaneAxis;
}

/** A box's size ON `axis` — the one dimension the anchor makes load-bearing. */
export const sizeOn = (axis: PaneAxis, b: { width: number; height: number }): number =>
	axis === 'y' ? b.height : b.width;

/**
 * Add an Oscillator and select it, so the inspector slides in. Returns its uid.
 *
 * `.open`, not merely visible: the pane is MOUNTED at every moment and parked off-edge until a
 * selection lands, so `toBeVisible` resolves on a pane that is still off-screen.
 */
export async function openInspector(page: Page): Promise<string> {
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(pane(page), 'a single selection opens the inspector').toHaveClass(/open/);
	return uid;
}

/** Hand the patch back: deselect, then remove the node the spec added. */
export async function dropNode(page: Page, uid: string): Promise<void> {
	await page.evaluate(() => (window as any).goofi.commands.clearSelection());
	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid).catch(() => {});
}

/** The resting grabber, normalised to the anchored axis — everything about it that MODALITY is
 *  supposed to decide, and nothing the anchor is. */
export interface Grabber {
	painted: boolean;
	/**
	 * Its length along the seam: `'seam'` when it runs the whole of it, otherwise the px it was
	 * given.
	 *
	 * Expressed this way because a seam's own length is the ANCHOR's fact — a sheet is as wide as
	 * its host, a right-hand pane as tall — while a pill's length is MODALITY's, and only the second
	 * is a number the two anchors have to agree on. Comparing raw px would have made a hairline in
	 * one anchor differ from the identical hairline in the other for a reason that is not the
	 * affordance at all.
	 */
	length: number | 'seam';
	/** Its thickness across the seam, in px. */
	across: number;
	rounded: boolean;
}

export async function grabberShape(page: Page): Promise<Grabber> {
	const axis = await paneAxis(page);
	return grip(page).evaluate((el, a) => {
		const cs = getComputedStyle(el, '::after');
		const w = parseFloat(cs.width);
		const h = parseFloat(cs.height);
		const seam = el.getBoundingClientRect();
		const along = a === 'y' ? w : h;
		const spans = Math.abs(along - (a === 'y' ? seam.width : seam.height)) < 1;
		return {
			painted: cs.backgroundColor !== 'rgba(0, 0, 0, 0)' && cs.backgroundColor !== 'transparent',
			length: spans ? ('seam' as const) : along,
			across: a === 'y' ? h : w,
			rounded: parseFloat(cs.borderTopLeftRadius) > 2
		};
	}, axis);
}

/**
 * The grabber's shape once it has stopped animating — two consecutive reads that agree.
 *
 * `::after` TRANSITIONS its background, so a one-shot read can describe a frame of that transition
 * rather than the resting state; and `toHaveCSS` cannot address a pseudo-element, so this is its
 * retrying equivalent.
 */
export async function settledGrabber(page: Page): Promise<Grabber> {
	let prev: Grabber | null = null;
	await expect
		.poll(
			async () => {
				const now = await grabberShape(page);
				const same = prev !== null && JSON.stringify(prev) === JSON.stringify(now);
				prev = now;
				return same;
			},
			{ message: 'the grabber settled' }
		)
		.toBe(true);
	return prev!;
}

/**
 * D-I9, as MODALITY decides it: a pointer with no hover gets a resting pill, and the SAME pill in
 * either anchor — 2.5rem along the seam, a hairline across it, pill-capped, painted at rest.
 *
 * The pill used to be declared by the portrait branch instead, which handed a phone two different
 * affordances for one finger (a chunky pill up, a thin line on its side) and handed a narrow docked
 * desktop column the touch grabber under a mouse. Asserting absolute numbers here — not "a pill
 * shape" — is what makes the two projects able to disagree.
 */
export async function expectRestingPill(page: Page): Promise<void> {
	const axis = await paneAxis(page);
	const rem = await rootRem(page);
	await expect
		.poll(() => grabberShape(page), {
			message: `a coarse pointer rests a 2.5rem pill on the ${axis} anchor`
		})
		.toEqual({ painted: true, length: expect.closeTo(2.5 * rem, 0), across: 2, rounded: true });
}

/**
 * Carry the grip `travel` px INTO the pane on its anchored axis, with a real finger.
 *
 * Into the pane is the POSITIVE direction on both axes — right for the right-hand pane, down for
 * the bottom sheet — which is exactly why `paneDrag`'s arithmetic needs no axis sign.
 *
 * Real CDP touch, not `page.mouse`: under `hasTouch` Playwright's mouse API still reports
 * `pointerType: 'mouse'`, which would prove the desktop path a second time.
 */
export async function swipeGripInward(page: Page, travel: number, steps = 8): Promise<void> {
	const axis = await paneAxis(page);
	const g = (await grip(page).boundingBox())!;
	const x0 = Math.round(g.x + g.width / 2);
	const y0 = Math.round(g.y + g.height / 2);
	const at = (d: number) =>
		axis === 'y' ? { x: x0, y: Math.round(y0 + d) } : { x: Math.round(x0 + d), y: y0 };
	const touch = await touchSession(page);
	await touch.down(at(0));
	for (let i = 1; i <= steps; i++) await touch.moveTo(at((travel * i) / steps));
	await touch.up();
}

/**
 * The pane's CEILING on `axis`, which only the stylesheet can evaluate (D-I6) — read off the
 * RESTING pane rather than re-derived here.
 *
 * With nothing stored the pane rests at the CSS fallback (`420px` / `60%`), and both phone
 * geometries cap below that, so the resting size IS the ceiling. Asserting the store is empty first
 * is what makes that inference sound instead of lucky.
 */
async function restingCeiling(page: Page, axis: PaneAxis): Promise<number> {
	for (const a of ['x', 'y'] as const)
		expect(
			await page.evaluate((k) => localStorage.getItem(k), PANE_AXES[a].key),
			`nothing is stored on ${a} yet, so the pane rests at its cap`
		).toBeNull();
	return sizeOn(axis, await settledBox(pane(page)));
}

/**
 * D-I4's UN-GATED half, in whichever anchor: the edge drag is THE resize, the same gesture on a
 * finger as on a mouse, over arithmetic that takes no axis at all. This is the finger;
 * `inspector-orientation.spec.ts` runs the mouse.
 *
 * Everything asserted is true in both anchors, which is the point of it being one function: a drag
 * inside the pane's own room lands it where the pointer asked, never dismisses it, leaves the other
 * dimension alone, and persists the RENDERED size under this axis's own key while never writing the
 * other's.
 *
 * `want` is what the drag ASKS for; the travel is clamped to the room the pane actually has
 * (`ceiling − floor`), which is the one quantity here that is NOT the same in the two anchors — not
 * because the gesture differs but because the two bounds are declared in different places and can
 * cross. On a landscape phone they do: `max-width: min(30%, 30rem)` is ~256px of an ~854px host,
 * just under the 260px floor, so the pane has NO room at all and the same drag that resizes the
 * sheet can only bottom the pane out or throw it away. That is a live D-I6 question — the cap and
 * the floor are both numbers the user set — and clamping here is what keeps this file measuring the
 * GESTURE rather than silently re-measuring that conflict.
 */
export async function expectEdgeDragFollowsThePointer(page: Page, want: number): Promise<void> {
	const axis = await paneAxis(page);
	const other = axis === 'y' ? 'x' : 'y';
	const { min, key } = PANE_AXES[axis];
	const before = await settledBox(pane(page));
	const ceiling = await restingCeiling(page, axis);
	const travel = Math.min(want, Math.max(ceiling - min, 0));

	await swipeGripInward(page, travel);

	const after = await settledBox(pane(page));
	expect(pane(page), 'a drag inside the pane’s own room is a resize, never a dismiss').toHaveClass(
		/open/
	);
	expect(
		sizeOn(axis, after),
		`${axis}: asked for ${ceiling - travel} (floor ${min}, ceiling ${ceiling})`
	).toBeCloseTo(Math.min(Math.max(ceiling - travel, min), ceiling), 0);
	expect(
		sizeOn(other, after),
		'and the other dimension is the anchor’s — the drag never touches it'
	).toBeCloseTo(sizeOn(other, before), 0);
	expect(
		await page.evaluate((k) => localStorage.getItem(k), key),
		'the RENDERED size is what was stored, so a reload agrees with the screen'
	).toBe(String(Math.round(sizeOn(axis, after))));
	expect(
		await page.evaluate((k) => localStorage.getItem(k), PANE_AXES[other].key),
		'…under this axis’s own key: one idiom, two keys, and the other is never written (D-I3)'
	).toBeNull();
}

/**
 * D-I4's MODALITY-GATED half, in whichever anchor: the same drag, carried a full overshoot past the
 * floor, throws the pane away instead of resizing it. Nothing is persisted — a dismiss is not a
 * resize, so the pane comes back the size it was.
 *
 * The swipe is the ONE thing input modality gates in this whole pane, and it is axis-blind by
 * construction (`endsInDismiss` reads a pointer type and a number, never an anchor). Proving it in
 * both orientations off one function is what keeps that true.
 */
export async function expectSwipeDismisses(page: Page): Promise<void> {
	const axis = await paneAxis(page);
	const { min, key } = PANE_AXES[axis];
	const ceiling = await restingCeiling(page, axis);
	// As far INTO the pane as the screen allows. Clamped to the viewport because a CDP touch point
	// off-screen is not a gesture any finger could make — and the assertion below is what proves the
	// room left is still more than a dismiss needs, rather than assuming it.
	const g = (await grip(page).boundingBox())!;
	const vp = page.viewportSize()!;
	const from = axis === 'y' ? g.y + g.height / 2 : g.x + g.width / 2;
	const travel = Math.floor((axis === 'y' ? vp.height : vp.width) - 2 - from);
	expect(travel, 'the swipe clears the floor by more than a full overshoot').toBeGreaterThan(
		ceiling - min + DISMISS_OVERSHOOT_PX
	);

	await swipeGripInward(page, travel);

	// Dismissing turns the inspector OFF, which unmounts the pane — the canvas is genuinely handed
	// back, exactly as the ✕ hands it back.
	await expect(pane(page), 'the swipe threw the pane away').toHaveCount(0);
	expect(
		await page.evaluate((k) => localStorage.getItem(k), key),
		'and a dismiss is not a resize: nothing was persisted, so it comes back its old size'
	).toBeNull();
}
