import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

/**
 * The viewer paint cap (Phil, 2026-08-08): the frontend paints at most `MAX_VIEWER_FPS` (30),
 * app-wide, however fast the producer runs or the display refreshes. Nodes update at ≤30 Hz in
 * practice; an uncapped rAF flush painted at display rate — 120 fps on a high-refresh phone —
 * re-painting frames that carried no new update.
 *
 * Driven through the real pipeline: an Oscillator pushed to 120 updates/s, its viewer subscribed,
 * and the perf HUD read after the meter settles. The bound is generous (≤ 40 against a 30 cap)
 * because the HUD's meter is a 500ms window — but an UNCAPPED flush in this harness paints at
 * ~60 (headless vsync), so the bound still separates the two behaviours cleanly.
 */
test('the viewer paint rate stays at the cap, however fast the producer runs', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [60, 60]);
	try {
		await waitForNode(page, uid);
		await page.evaluate(
			(u) => (window as any).goofi.commands.updateParam(u, 'common', 'max_frequency', 120),
			uid
		);
		// Let the new rate flow and the HUD's 500ms window fill.
		await expect(page.getByTestId('perf-hud')).toBeAttached();
		await page.waitForTimeout(1500);
		const fps = await page
			.getByTestId('perf-hud')
			.locator('.fps')
			.evaluate((el) => parseFloat(el.textContent ?? '0'));
		expect(fps, 'frames flow at all (the producer really runs)').toBeGreaterThan(15);
		expect(fps, 'painted at the cap, not at the display or producer rate').toBeLessThanOrEqual(40);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
	}
});
