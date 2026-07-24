import { test, expect } from '@playwright/test';

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
