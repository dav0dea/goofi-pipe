import { expect, type Locator, type Page } from '@playwright/test';
import { settledBox } from './geometry';
import { addNode, selectNode, waitForNode, waitForNoNode } from './goofi';
import { touchSession } from './touch';

/**
 * The inspector pane, and THE ONE RULE it is built on:
 *
 *   ORIENTATION decides the anchored AXIS — portrait a bottom sheet, landscape/desktop the
 *   right-hand edge. INPUT MODALITY decides only the resting AFFORDANCE: the grabber pill and the
 *   coarse hit band. THE GESTURE IS UNIFORM — edge-drag, identical on a mouse and a finger — and
 *   closing is the explicit ✕ alone.
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
 * Where each anchor's size is remembered, mirroring `frontend/src/lib/panels/paneDrag.ts` — the e2e
 * package cannot import the app's source, so a `localStorage` key has to be restated somewhere, and
 * this is the one place.
 *
 * Neither of the pane's BOUNDS is here any more: they are one host-relative `clamp()` per axis in
 * the stylesheet, and `paneBound` below asks CSS for them. Restating the floor was how this file
 * came to hold a `260` that the landscape ceiling had already fallen under.
 */
export const PANE_KEYS: Record<PaneAxis, string> = {
	x: 'goofi.panelWidth',
	y: 'goofi.panelHeight'
};

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

/**
 * The pane's FLOOR or its CEILING on whichever axis it is anchored on — ASKED OF CSS, never
 * restated here.
 *
 * Both bounds are one `clamp()` per axis in the stylesheet and both are HOST-relative, so only the
 * browser can evaluate them; this file used to hold its own copy of the floor, which is how it came
 * to assert a 260 the landscape ceiling had already fallen under. Driving the pane's own size
 * variable to either extreme and measuring what comes back asks the question CSS answers on every
 * frame, so a retuned limit moves this reading with it rather than against it.
 *
 * One `evaluate`, because `getBoundingClientRect` forces layout synchronously: the drive, the read
 * and the restore are a single task and no frame ever paints the pane at an extreme.
 */
export async function paneBound(page: Page, which: 'floor' | 'ceiling'): Promise<number> {
	const axis = await paneAxis(page);
	return pane(page).evaluate(
		(el, [a, drive]) => {
			const prop = a === 'y' ? '--pane-h' : '--pane-w';
			const had = el.style.getPropertyValue(prop);
			el.style.setProperty(prop, drive);
			const r = el.getBoundingClientRect();
			const size = a === 'y' ? r.height : r.width;
			if (had) el.style.setProperty(prop, had);
			else el.style.removeProperty(prop);
			return size;
		},
		[axis, which === 'floor' ? '0px' : '1000000px'] as const
	);
}

/** A box's size ON `axis` — the one dimension the anchor makes load-bearing. */
export const sizeOn = (axis: PaneAxis, b: { width: number; height: number }): number =>
	axis === 'y' ? b.height : b.width;

/**
 * Add an Oscillator and select it, so the inspector slides in. Returns its uid.
 *
 * `.open`, not merely attached: the pane is MOUNTED at every moment and hidden off-edge until a
 * selection lands, so attachment alone does not say that the slide has started.
 */
export async function openInspector(page: Page): Promise<string> {
	const uid = await addNode(page, 'Oscillator', [40, 40]);
	await waitForNode(page, uid);
	await selectNode(page, uid);
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
 *
 * THE FINGER COMES TO REST BEFORE IT LIFTS, and that is not decoration. Dispatched back to back, the
 * moves carry a velocity Chromium reads as a FLING at release — and Chromium eats the next tap
 * anywhere on the page to stop the fling it thinks is running (measured: after this drag, a tap on
 * bare canvas far from the pane also did nothing; the same drag with a still finger before release
 * left the tap intact). No product code is involved: the sequence still arrives complete and
 * unprevented, Chromium simply synthesizes no `click` from it. A deliberate resize ends with a
 * finger that has stopped, so holding still is also the more faithful gesture.
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
	await touch.moveTo(at(travel));
	await page.waitForTimeout(150);
	await touch.up();
}

/**
 * The pane's RESTING size on `axis` — the middle term of the stylesheet's `clamp()`, measured off
 * the pane rather than re-derived here.
 *
 * It is only the resting size while nothing is stored, since a dragged size replaces it; asserting
 * the store is empty first is what makes that sound instead of lucky. It used to be the CEILING as
 * well, because the old cap resolved below the fallback on every geometry the app had met — the
 * limits are 10 %–90 % now, so the resting size sits strictly inside them and the two are different
 * numbers.
 */
async function restingSize(page: Page, axis: PaneAxis): Promise<number> {
	for (const a of ['x', 'y'] as const)
		expect(
			await page.evaluate((k) => localStorage.getItem(k), PANE_KEYS[a]),
			`nothing is stored on ${a} yet, so the pane is at its resting size`
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
 * `want` is dragged in FULL, and the room to do it in is asserted rather than clamped to. It used
 * to be clamped — `min(want, max(ceiling − floor, 0))` — which on a landscape phone, where the two
 * bounds had crossed, quietly made this a ZERO-pixel drag: every assertion below then held
 * trivially, so the one project that could have reported the defect reported a pass instead. A test
 * that measures a gesture must not silently stop measuring it.
 */
export async function expectEdgeDragFollowsThePointer(page: Page, want: number): Promise<void> {
	const axis = await paneAxis(page);
	const other = axis === 'y' ? 'x' : 'y';
	const floor = await paneBound(page, 'floor');
	const key = PANE_KEYS[axis];
	const before = await settledBox(pane(page));
	const resting = await restingSize(page, axis);
	expect(
		resting - floor,
		`${axis}: the pane has room for the ${want}px asked of it (floor ${floor}, resting ${resting})`
	).toBeGreaterThanOrEqual(want);

	await swipeGripInward(page, want);

	const after = await settledBox(pane(page));
	expect(pane(page), 'a drag inside the pane’s own room lands where it was asked').toHaveClass(
		/open/
	);
	expect(
		sizeOn(axis, after),
		`${axis}: asked for ${resting - want} (floor ${floor}, resting ${resting})`
	).toBeCloseTo(resting - want, 0);
	expect(
		sizeOn(other, after),
		'and the other dimension is the anchor’s — the drag never touches it'
	).toBeCloseTo(sizeOn(other, before), 0);
	expect(
		await page.evaluate((k) => localStorage.getItem(k), key),
		'the RENDERED size is what was stored, so a reload agrees with the screen'
	).toBe(String(Math.round(sizeOn(axis, after))));
	expect(
		await page.evaluate((k) => localStorage.getItem(k), PANE_KEYS[other]),
		'…under this axis’s own key: one idiom, two keys, and the other is never written (D-I3)'
	).toBeNull();
}

/**
 * Carry the grip as far INTO the pane as the screen allows, and hand back the floor CSS bottomed it
 * out at — the pane at its smallest legal size, in whichever anchor.
 *
 * The travel is clamped to the viewport because a CDP touch point off-screen is not a gesture any
 * finger could make, and asserted against the room the pane actually has, so a drag that merely
 * reached the floor cannot pass for one that cleared it.
 *
 * That the pane is still OPEN afterwards is asserted here rather than left to the callers: this is
 * exactly where a swipe-past-the-floor used to throw it away, the one thing input modality ever
 * gated in this pane. It is gone, and stating its absence at the moment the drag resolves is what
 * makes re-adding it fail loudly instead of turning every later assertion into a missing element.
 */
export async function dragGripToTheFloor(page: Page): Promise<number> {
	const axis = await paneAxis(page);
	const floor = await paneBound(page, 'floor');
	const resting = await restingSize(page, axis);
	const g = (await grip(page).boundingBox())!;
	const vp = page.viewportSize()!;
	const from = axis === 'y' ? g.y + g.height / 2 : g.x + g.width / 2;
	const travel = Math.floor((axis === 'y' ? vp.height : vp.width) - 2 - from);
	expect(travel, 'the drag carries the pane past its floor, not merely down to it').toBeGreaterThan(
		resting - floor
	);

	await swipeGripInward(page, travel);

	await expect(pane(page), 'a finger cannot throw the pane away — only the ✕ closes it').toHaveClass(
		/open/
	);
	expect(
		sizeOn(axis, await settledBox(pane(page))),
		`${axis}: bottomed out at the floor, not closed`
	).toBeCloseTo(floor, 0);
	return floor;
}

/**
 * The GESTURE IS UNIFORM, in whichever anchor: a drag carried past the floor is still a RESIZE, so
 * the floored size is persisted like any other. `inspector-orientation.spec.ts` runs the same drag
 * with a mouse; there is no pointer type for which it means something else.
 */
export async function expectDragPastTheFloorStillResizes(page: Page): Promise<void> {
	const key = PANE_KEYS[await paneAxis(page)];
	const floor = await dragGripToTheFloor(page);
	expect(
		await page.evaluate((k) => localStorage.getItem(k), key),
		'and it is a resize like any other, so the floored size is what was persisted'
	).toBe(String(Math.round(floor)));
}
