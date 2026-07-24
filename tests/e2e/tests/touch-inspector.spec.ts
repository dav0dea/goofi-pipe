import { test, expect, type Locator } from '@playwright/test';

// Runs ONLY under the `touch` project (Pixel 7 → (pointer:coarse)/(hover:none) true), where app.css
// floors input/select/textarea font-size to 16px to defeat iOS focus-zoom. Two inspector controls
// out-specified that floor with a sub-16px F token — the fx multi-line editor (--fs-micro ≈ 10px) and
// the header rename input (--fs-strong ≈ 14px on phone) — so a tap into either force-zoomed the
// viewport on focus. Each must now resolve to >= 16px under a coarse pointer (the desktop sub-16px
// sizes are kept — only the coarse-pointer override lifts them).
const fontPx = (loc: Locator): Promise<number> =>
	loc.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));

test('the fx multi-line editor is >= 16px under a coarse pointer (no iOS focus-zoom)', async ({
	page
}) => {
	await page.goto('/dev/inspector');
	const field = page.getByTestId('inspector-fx');
	// Enable fx, then grow the in-panel multi-line editor so the textarea renders.
	await field.getByTestId('param-fx-toggle').click();
	await field.getByTestId('param-expr-expand').click();
	const ta = field.getByTestId('param-expr-multiline');
	await expect(ta).toBeVisible();
	expect(await fontPx(ta), 'textarea font-size >= 16 defeats iOS focus-zoom').toBeGreaterThanOrEqual(16);
});

test('the header rename input is >= 16px under a coarse pointer (no iOS focus-zoom)', async ({
	page
}) => {
	await page.goto('/dev/inspector');
	// The ParamForm sample renders its identity header; open the inline rename to mount its input.
	const form = page.getByTestId('inspector-form');
	await form.getByTestId('node-name').click();
	const input = form.getByTestId('node-name-input');
	await expect(input).toBeVisible();
	expect(await fontPx(input), 'rename input font-size >= 16 defeats iOS focus-zoom').toBeGreaterThanOrEqual(16);
});
