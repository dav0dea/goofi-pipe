import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import {
	DISMISS_OVERSHOOT_PX,
	dropNode,
	editorHost,
	expectEdgeDragFollowsThePointer,
	expectRestingPill,
	expectSwipeDismisses,
	openInspector,
	paneAxis,
	paneMin,
	paneRoom
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

/* …and the second anti-vacuity guard, which is the one that was missing: two anchors running one
   set of assertions proves nothing about an anchor where there is nothing left to assert. On THIS
   project's 854px host the pane's two bounds crossed — a 256px ceiling under a 260px floor — so the
   landscape pane had NEGATIVE room, the edge drag below was dragging zero pixels and passing, and a
   ~40px slip of a finger was already a dismiss. `--pane-min` makes an empty range unspellable; this
   is what says so out loud, per anchor, if a future edit to either bound crosses them again. */
test('the pane has real room between its floor and its ceiling', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		const room = await paneRoom(page);
		expect(
			room,
			`${await paneAxis(page)}: ceiling − floor, with the floor at ${await paneMin(page)}`
		).toBeGreaterThan(0);
		// And more than a nudge: a pane whose entire range is shorter than the overshoot a dismiss
		// has to clear is not resizable, it is a throw-away with a preamble.
		expect(room, 'and more room than a dismiss has to be carried past').toBeGreaterThan(
			DISMISS_OVERSHOOT_PX
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

test('a swipe carried past the floor dismisses the pane (D-I4)', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		await expectSwipeDismisses(page);
	} finally {
		// The dismiss flipped this editor's inspector off, which is per-PAGE state (`selection`'s
		// `inspectorOn`, never the layout blob) — so the next spec's fresh page has it on again and
		// there is nothing to hand back but the node.
		await dropNode(page, uid);
	}
});
