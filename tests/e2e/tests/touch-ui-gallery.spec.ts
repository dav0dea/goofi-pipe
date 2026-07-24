import { test, expect } from '@playwright/test';

// Runs ONLY under the `touch` project (Pixel 7 emulation → hasTouch+isMobile flip
// (pointer:coarse)/(hover:none) true), so app.css floors --hit to 44px. The IconButton's
// rendered box must therefore be a real 44px tap target in BOTH dimensions while its glyph
// element stays visually small — the deferred-from-F icon-control hit-targeting requirement.
test('IconButton meets the 44px coarse tap target while its glyph stays small', async ({ page }) => {
	await page.goto('/dev/ui');
	const btn = page.getByTestId('ui-icon-primary-md');
	await btn.waitFor();

	const box = await btn.boundingBox();
	expect(box, 'the IconButton has a rendered box').not.toBeNull();
	expect(box!.width, 'coarse tap-target width >= 44').toBeGreaterThanOrEqual(44);
	expect(box!.height, 'coarse tap-target height >= 44').toBeGreaterThanOrEqual(44);

	// The glyph itself must NOT be inflated to the tap target — it stays small and centered.
	const glyph = btn.locator('.glyph');
	const gbox = await glyph.boundingBox();
	expect(gbox, 'the glyph has a rendered box').not.toBeNull();
	expect(gbox!.width, 'the glyph stays small (not the tap target)').toBeLessThan(24);
	expect(gbox!.height, 'the glyph stays small (not the tap target)').toBeLessThan(24);
});
