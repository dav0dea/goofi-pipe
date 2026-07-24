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

// The Field-family controls must also honour the coarse --hit floor: a NumberInput inherits the
// app.css `min-height: var(--hit)` (44px under a coarse pointer), and the Toggle sizes its box to
// --hit. Both must be real tap targets on touch — the Field family is what R renders on phones.
test('Field controls meet the 44px coarse tap target', async ({ page }) => {
	await page.goto('/dev/ui');

	const num = page.getByTestId('ui-field-number');
	await num.waitFor();
	const nbox = await num.boundingBox();
	expect(nbox, 'the NumberInput has a rendered box').not.toBeNull();
	expect(nbox!.height, 'NumberInput height >= 44 under coarse').toBeGreaterThanOrEqual(44);

	const toggle = page.getByTestId('ui-toggle');
	await toggle.waitFor();
	const tbox = await toggle.boundingBox();
	expect(tbox, 'the Toggle has a rendered box').not.toBeNull();
	expect(tbox!.height, 'Toggle height >= 44 under coarse').toBeGreaterThanOrEqual(44);
});

// Tabs and Disclosure are chrome R renders on phones — each tab and the disclosure summary must be a
// real 44px tap target under a coarse pointer (both size their control to var(--hit)).
test('Tabs + Disclosure meet the 44px coarse tap target', async ({ page }) => {
	await page.goto('/dev/ui');

	const tab = page.getByTestId('ui-tabs').getByRole('tab').first();
	await tab.waitFor();
	const tabBox = await tab.boundingBox();
	expect(tabBox, 'the tab has a rendered box').not.toBeNull();
	expect(tabBox!.height, 'tab height >= 44 under coarse').toBeGreaterThanOrEqual(44);

	const summary = page.getByTestId('ui-disclosure').getByRole('button');
	await summary.waitFor();
	const sbox = await summary.boundingBox();
	expect(sbox, 'the disclosure summary has a rendered box').not.toBeNull();
	expect(sbox!.height, 'disclosure summary height >= 44 under coarse').toBeGreaterThanOrEqual(44);
});
