import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import {
	dropNode,
	editorHost,
	expectRestingPill,
	expectSwipeDismisses,
	openInspector,
	paneAxis
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

test('a coarse pointer rests the SAME grabber pill in either anchor (D-I9)', async ({ page }) => {
	const uid = await openInspector(page);
	try {
		await expectRestingPill(page);
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
