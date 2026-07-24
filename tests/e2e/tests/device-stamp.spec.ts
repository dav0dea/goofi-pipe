import { test, expect } from '@playwright/test';

// The device seam stamps <html> so global CSS / SvelteFlow overrides (outside any component)
// can read the device class. Mirrors the data-theme precedent.
test('the device store stamps data-* on <html>', async ({ page }) => {
	await page.goto('/');
	const html = page.locator('html');
	await expect(html).toHaveAttribute('data-pointer', /coarse|fine/);
	await expect(html).toHaveAttribute('data-size', /phone|compact|full/);
});
