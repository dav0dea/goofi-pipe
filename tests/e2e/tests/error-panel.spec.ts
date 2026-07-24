import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

// M-Task 2 gave the ErrorPanel the real dismissal it never had: the error dropdown used to only
// toggle from the chip — no Escape, no outside-click. It now delegates to the Popover primitive,
// so this asserts the NEW behaviour on the REAL panel, driven by a REAL node error.
//
// The error is reachable and environment-independent: the tier calls `node.process(**present)` with
// only the CONNECTED inputs as kwargs (goofi-pymod exec.rs `run_process`), and `LempelZiv.process`
// has a required `data` param — so a node added with nothing connected raises a missing-argument
// TypeError on its first tick. That is a per-tick Python-node error, exactly the "a node that raises"
// case the panel exists to surface (it sets the same `error` field that drives the floating chip).
// It does NOT depend on any missing dependency (numpy is present in both venvs) — it is the
// unconnected-required-input raise, which is permanent. The node is torn down after each test so the
// shared backend graph stays clean for later specs.
test.describe('ErrorPanel dismissal (delegated to Popover)', () => {
	let created: string[] = [];

	test.afterEach(async ({ page }) => {
		for (const uid of created) {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
		created = [];
	});

	async function summonErrorChip(page: Page) {
		await page.goto('/');
		await waitForApp(page);
		// A Python node whose process() needs a connected ARRAY input; added with none, it raises on
		// its first tick and the error surfaces as the floating chip.
		const uid = await addNode(page, 'LempelZiv', 'python');
		created.push(uid);
		await waitForNode(page, uid);
		const chipHost = page.getByTestId('error-chip');
		await expect(chipHost, 'a real node error raises the floating error chip').toBeVisible();
		return chipHost;
	}

	test('the chip opens the error popover, and Escape dismisses it', async ({ page }) => {
		const chipHost = await summonErrorChip(page);
		const popover = page.getByTestId('error-popover');

		await expect(popover, 'closed by default').toBeHidden();
		await chipHost.locator('button').click();
		await expect(popover, 'clicking the chip opens the popover').toBeVisible();

		await page.keyboard.press('Escape');
		await expect(popover, 'Escape dismisses the popover (new behaviour)').toBeHidden();
	});

	test('an outside click dismisses the error popover', async ({ page }) => {
		const chipHost = await summonErrorChip(page);
		const popover = page.getByTestId('error-popover');

		await chipHost.locator('button').click();
		await expect(popover).toBeVisible();
		// Click the top-left of the viewport — neither the popover surface nor the chip anchor.
		await page.mouse.click(10, 10);
		await expect(popover, 'an outside click dismisses the popover (new behaviour)').toBeHidden();
	});
});
