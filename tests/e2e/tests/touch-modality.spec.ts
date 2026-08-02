import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { settledBox } from '../lib/geometry';
import {
	dropNode,
	editorHost,
	expectDragPastTheFloorStillResizes,
	expectEdgeDragFollowsThePointer,
	expectRestingPill,
	openInspector,
	pane,
	paneAxis,
	paneBound,
	rootRem,
	sizeOn
} from '../lib/inspector';

/**
 * The inspector's INPUT-MODALITY behaviour, run in BOTH orientations off ONE set of assertions.
 *
 * `playwright.config.ts` points `touch` (Pixel 7 portrait → the bottom sheet) and `touch-landscape`
 * (→ the right-hand pane) at this file. Everything asserted below lives in `lib/inspector.ts`,
 * normalised to whichever axis the container query anchored the pane on, so the two projects run
 * the same bytes rather than two copies that can drift apart.
 *
 * That IS the guard. The rule is that orientation decides only the axis and modality decides the
 * gesture and its affordance; a change that re-couples them makes exactly one of these projects
 * disagree, immediately and by name. It is what caught the defect this file was written for: a
 * resting grabber declared by the portrait branch, so one finger got a chunky pill standing up and
 * a thin line lying down.
 *
 * Only two projects, not three: `tablet` is portrait as well, so it would re-measure `touch`'s
 * answer. What has to be re-run is the OTHER anchor, and there is exactly one of those.
 */

test.beforeEach(async ({ page }) => {
	// The pane slides over `--dur-slow` and nearly everything here is a POSITION; the app's own
	// reduced-motion rule collapses the transition, so a read describes the layout and not a frame.
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
});

/* The anti-vacuity guard, and it comes first: "the same assertions in both orientations" says
   nothing unless the two projects really are in different anchors. The anchor is asked of the HOST
   PANEL's shape (D-I2) — not the viewport, not the device class — so this asserts the pane against
   the very box the container query measures. */
test('this project’s host panel picks the anchor, and the pane agrees with it', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		const host = (await editorHost(page).boundingBox())!;
		expect(await paneAxis(page), 'the anchor is the host panel’s shape and nothing else').toBe(
			host.height >= host.width ? 'y' : 'x'
		);
	} finally {
		await dropNode(page, uid);
	}
});

/* The limits, in whichever anchor: a TENTH of the host at the bottom, NINE TENTHS at the top.
   Measured off the pane rather than restated, and measured against the HOST panel rather than the
   viewport — the pane answers to the surface it lives in (D-I2), and on this phone the two are
   different numbers. One `clamp()` per axis states both, so the floor is below the ceiling by
   construction and the pair that crossed (a 256px ceiling under a 260px floor, an EMPTY range) is
   no longer spellable. */
test('the pane’s floor and its ceiling are a tenth and nine tenths of its HOST', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		const axis = await paneAxis(page);
		const host = sizeOn(axis, (await editorHost(page).boundingBox())!);
		expect(await paneBound(page, 'floor'), `${axis}: a tenth of a ${host}px host`).toBeCloseTo(
			host * 0.1,
			0
		);
		expect(
			await paneBound(page, 'ceiling'),
			`${axis}: nine tenths of a ${host}px host`
		).toBeCloseTo(host * 0.9, 0);
	} finally {
		await dropNode(page, uid);
	}
});

/* …and widening them MOVED NOTHING. The resting size used to EMERGE from the old ceiling clamping
   a larger fallback, so raising the ceiling is precisely the edit that could have dragged it along;
   it is stated outright now, and this is what says it did not move.

   Sitting strictly INSIDE both limits is the other half of the same claim, and the ANTI-VACUITY
   guard the rest of this file rests on: two anchors running one set of assertions prove nothing
   about an anchor where there is nothing left to assert. On THIS project's 854px host the pane's
   two bounds once crossed — a 256px ceiling under a 260px floor — so the landscape pane had
   NEGATIVE room and the edge drag below was dragging zero pixels and passing. */
test('the resting default did not move, and now sits strictly inside those limits', async ({
	page
}) => {
	const uid = await openInspector(page);
	try {
		const axis = await paneAxis(page);
		const host = (await editorHost(page).boundingBox())!;
		const resting = sizeOn(axis, await settledBox(pane(page)));
		expect(
			resting,
			axis === 'y' ? '60% of the host' : 'the lesser of 40% of the host and 30rem'
		).toBeCloseTo(
			axis === 'y' ? host.height * 0.6 : Math.min(host.width * 0.4, 30 * (await rootRem(page))),
			0
		);
		expect(resting, 'above the floor, so it can be dragged smaller').toBeGreaterThan(
			await paneBound(page, 'floor')
		);
		expect(resting, 'and below the ceiling, so it can be dragged larger').toBeLessThan(
			await paneBound(page, 'ceiling')
		);
	} finally {
		await dropNode(page, uid);
	}
});

test('a coarse pointer rests the SAME grabber pill in either anchor (D-I9)', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		await expectRestingPill(page);
	} finally {
		await dropNode(page, uid);
	}
});

/* The other half of D-I4's "edge drag is for BOTH", and the half that was only ever proved on the Y
   axis: the same grip, the same module, the same persistence idiom, a finger instead of a mouse.
   `inspector-orientation.spec.ts` runs the mouse in both anchors; this runs the finger in both. */
test('an edge drag by TOUCH moves the pane to where the pointer asked (D-I3/D-I4)', async ({
	page
}) => {
	const uid = await openInspector(page);
	try {
		// 60px: inside the room the TIGHTER anchor has (landscape's 82px above its floor), because
		// the helper now asserts that room rather than clamping the drag down into it.
		await expectEdgeDragFollowsThePointer(page, 60);
	} finally {
		await dropNode(page, uid);
	}
});

test('a drag carried past the floor is still a resize, never a dismiss (D-I4)', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		await expectDragPastTheFloorStillResizes(page);
	} finally {
		await dropNode(page, uid);
	}
});
