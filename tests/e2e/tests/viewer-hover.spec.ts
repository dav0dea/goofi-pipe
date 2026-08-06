import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import {
	VIEWER_HOVER_SURFACES,
	hoverSettled,
	surfaceStyles,
	unhover
} from '../lib/viewerChrome';

/**
 * The fine-pointer half of `touch-viewer.spec.ts`, over the same list of surfaces.
 *
 * Gating the viewer's five `:hover` rules on `(hover: hover)` is a change a mouse must not be able
 * to see, and "the existing specs still pass" does not prove that: none of them reads a hover
 * colour. This does — every surface the coarse spec proves stays at rest is proved here to still
 * move.
 */
test('every viewer hover rule still answers a mouse', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	const slot = (p: Page) => p.locator(`.slot-viewer[data-node="${uid}"]`);
	await expect(slot(page)).toBeVisible();
	try {
		await unhover(page);
		const rest = new Map<string, string[]>();
		for (const s of VIEWER_HOVER_SURFACES) rest.set(s.name, await surfaceStyles(page, uid, s));

		for (const s of VIEWER_HOVER_SURFACES) {
			await hoverSettled(page, uid, s);
			const now = await surfaceStyles(page, uid, s);
			expect(now, `${s.name} still lights up under the mouse`).not.toEqual(rest.get(s.name));
			await unhover(page);
		}
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});
