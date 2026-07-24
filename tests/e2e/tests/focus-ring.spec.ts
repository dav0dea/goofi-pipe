import { test, expect } from '@playwright/test';

// A keyboard-driven tool must show focus. Before F there were zero :focus-visible rules.
// Drive an actual keyboard Tab — programmatic .focus() does not reliably trigger :focus-visible,
// but keyboard focus does, and our universal `:focus-visible` rule rings whatever gets focused.
test('a keyboard-focused element shows a visible focus ring', async ({ page }) => {
	await page.goto('/');
	await page.locator('button').first().waitFor(); // app hydrated, something is focusable
	await page.keyboard.press('Tab');
	const ring = await page.evaluate(() => {
		const el = document.activeElement as HTMLElement | null;
		if (!el || el === document.body) return { focused: false, outlineStyle: 'none' };
		const s = getComputedStyle(el);
		return { focused: true, outlineStyle: s.outlineStyle };
	});
	expect(ring.focused, 'Tab must move focus into the app').toBe(true);
	expect(ring.outlineStyle, 'a keyboard-focused element must render a visible outline').not.toBe('none');
});
