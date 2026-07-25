import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

// The device seam stamps <html> so global CSS / SvelteFlow overrides (outside any component)
// can read the device class.
test('the device store stamps data-* on <html>', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const html = page.locator('html');
	await expect(html).toHaveAttribute('data-pointer', /coarse|fine/);
	await expect(html).toHaveAttribute('data-size', /phone|compact|full/);
});

// stamp() also publishes --kb-inset (setProperty) and toggles data-short (toggleAttribute).
// At the default desktop viewport (height > 480, no soft keyboard) both must read their
// no-keyboard / not-short values — this guards the kbInset() arg order and the height gate.
test('the device store publishes --kb-inset and data-short on <html>', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const kbInset = await page.evaluate(() =>
		getComputedStyle(document.documentElement).getPropertyValue('--kb-inset').trim()
	);
	expect(kbInset).toBe('0px');
	const short = await page.evaluate(() => document.documentElement.hasAttribute('data-short'));
	expect(short).toBe(false);
});
