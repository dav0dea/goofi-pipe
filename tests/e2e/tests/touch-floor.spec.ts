import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

// Under a coarse pointer, interactive controls must meet a 44px touch target.
// Under a fine pointer (the default project) they must NOT be inflated — verified by the
// absence of this project from the default run (testMatch scopes it to the `touch` project).
test('coarse pointer floors control height at 44px', async ({ page }) => {
	await page.goto('/');
	const btn = page.locator('button', { hasText: /save/i }).first();
	await btn.waitFor();
	const h = await btn.evaluate((el) => el.getBoundingClientRect().height);
	expect(h).toBeGreaterThanOrEqual(44);
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
