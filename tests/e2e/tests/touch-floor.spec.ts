import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

// Under a coarse pointer, interactive controls must meet a 44px touch target.
// Under a fine pointer (the default project) they must NOT be inflated — verified by the
// absence of this project from the default run (testMatch scopes it to the `touch` project).
test('coarse pointer floors control height at 44px', async ({ page }) => {
	// Wide enough that the header's Save is still IN the bar. Since R-Task 6 the actions spill into
	// the overflow menu as the width runs out, and at the phone's own 412px every one of them has —
	// a spilled action is `display: none`, which has no box to measure. The question here is the
	// coarse floor, not how much bar a phone has (the collapse keys on width, not device class, so
	// this viewport is as coarse as the default one).
	await page.setViewportSize({ width: 1000, height: 800 });
	await page.goto('/');
	// By testid, not text: Save is an icon button now (the header's one icon family).
	const btn = page.getByTestId('topbar-save');
	await btn.waitFor();
	const h = await btn.evaluate((el) => el.getBoundingClientRect().height);
	expect(h).toBeGreaterThanOrEqual(44);
	const barH = await page
		.locator('.topbar')
		.evaluate((el) => el.getBoundingClientRect().height);
	expect(barH, 'the natural-height bar grows with its coarse-pointer controls').toBe(h);
});

// The workspace chrome strips render deliberately dense icon buttons — the tab bar's ＋ at 22px
// and the panel header's maximize/close at 20px, both under the IconButton `--hit` floor
// (workspace-chrome.spec pins those fine-pointer numbers). The dense box is a FINE-pointer
// affordance only: under a coarse pointer the floor must be restored so each is a real tap
// target. That restore is `density="chrome"` in the IconButton primitive; this is its guard —
// nothing else in the suite covers it, so a strip that re-pins its own box (or a primitive that
// drops the floor) would otherwise ship a 20px touch target silently.
test('the chrome-dense workspace icon buttons meet the 44px coarse tap target', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const add = page.getByTestId('workspace-tabs').getByRole('button', { name: 'New tab' });
	await add.waitFor();
	const abox = (await add.boundingBox())!;
	expect(abox.width, 'the tab-strip ＋ is a real tap target on touch').toBeGreaterThanOrEqual(44);
	expect(abox.height, 'the tab-strip ＋ is a real tap target on touch').toBeGreaterThanOrEqual(44);

	const max = page
		.getByTestId('panel-header')
		.first()
		.getByRole('button', { name: 'Maximize panel' });
	await max.waitFor();
	const mbox = (await max.boundingBox())!;
	expect(mbox.width, 'the header maximize is a real tap target on touch').toBeGreaterThanOrEqual(44);
	expect(mbox.height, 'the header maximize is a real tap target on touch').toBeGreaterThanOrEqual(
		44
	);
});

// The other end of the same rule. A slot header is FROZEN node-canvas geometry (`--node-u`, 24px,
// mirrored in nodeMetrics.ts), so it is the one strip that cannot grow to hold a floored box: the
// viewer-settings cog pins its VISUAL box under the floor and lets IconButton's coarse `::after`
// carry the tap target instead. Routing it through `density="chrome"` would look like the same
// seam the two buttons above use, and would silently stand a 44px cog inside a 24px header.
// Compared against the header rather than a literal, so canvas zoom cannot move the goalposts.
test('the viewer-settings cog stays inside the frozen slot header on touch', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);

	const header = page.locator(`.slot-viewer[data-node="${uid}"] header`).first();
	const cog = header.getByTestId('viewer-settings-cog');
	await cog.waitFor();
	const hbox = (await header.boundingBox())!;
	const cbox = (await cog.boundingBox())!;
	expect(cbox.height, 'the cog does not outgrow the frozen header it sits in').toBeLessThanOrEqual(
		hbox.height
	);

	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid);
});
