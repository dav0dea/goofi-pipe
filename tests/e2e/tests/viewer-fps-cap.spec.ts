import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

/**
 * The viewer paint cap (Phil, 2026-08-08): the frontend paints at most `MAX_VIEWER_FPS` (30),
 * app-wide, however fast the producer runs or the display refreshes. Nodes update at ≤30 Hz in
 * practice; an uncapped rAF flush painted at display rate — 120 fps on a high-refresh phone —
 * re-painting frames that carried no new update.
 *
 * Driven through the real pipeline: Oscillators with their viewers subscribed, and the perf HUD
 * read after the meter settles. The bound is generous (≤ 40 against a 30 cap) because the HUD's
 * meter is a 500ms window — but an UNCAPPED flush in this harness paints at ~60 (headless vsync),
 * so the bound still separates the two behaviours cleanly.
 *
 * Two axes, and the second one is why this file has a second node. The cap is app-wide: ONE rAF
 * flush repaints every stream that has a new frame, so the readout must not move with the number
 * of open streams either. Asserting that needed a TWO-stream fixture — with one node the paint
 * rate and the old per-slot SUM are the same number, so this spec sat green for two days while the
 * HUD climbed 30 fps per node added (`fps-counter-investigation.md`). A single-stream fixture
 * cannot express a summing bug.
 */
const readFps = (page: import('@playwright/test').Page): Promise<number> =>
	page
		.getByTestId('perf-hud')
		.locator('.fps')
		.evaluate((el) => parseFloat(el.textContent ?? '0'));

test('the viewer paint rate stays at the cap, however fast the producer runs', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uids = [await addNode(page, 'Oscillator', 'inputs', [60, 60])];
	try {
		await waitForNode(page, uids[0]);
		await page.evaluate(
			(u) => (window as any).goofi.commands.updateParam(u, 'common', 'max_frequency', 120),
			uids[0]
		);
		// Let the new rate flow and the HUD's 500ms window fill.
		await expect(page.getByTestId('perf-hud')).toBeAttached();
		await page.waitForTimeout(1500);
		const one = await readFps(page);
		expect(one, 'frames flow at all (the producer really runs)').toBeGreaterThan(15);
		expect(one, 'painted at the cap, not at the display or producer rate').toBeLessThanOrEqual(40);

		// The node-count axis: a second stream, painted by the same app-wide flush.
		uids.push(await addNode(page, 'Oscillator', 'inputs', [60, 300]));
		await waitForNode(page, uids[1]);
		await page.waitForTimeout(1500);
		const two = await readFps(page);
		expect(two, 'the second stream really paints too').toBeGreaterThan(15);
		expect(
			two,
			`the paint rate is one app-wide number, not a per-stream sum (1 node: ${one}, 2 nodes: ${two})`
		).toBeLessThanOrEqual(40);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes(u), uids);
	}
});
