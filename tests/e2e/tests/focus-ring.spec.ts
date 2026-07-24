import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

// A keyboard-driven tool must show focus. Before F there were zero :focus-visible rules.
// Drive an actual keyboard Tab — programmatic .focus() does not reliably trigger :focus-visible,
// but keyboard focus does, and our universal `:focus-visible` rule rings whatever gets focused.
test('a keyboard-focused element shows the app accent focus ring', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page); // deterministic hydration (same gate the other specs use)
	await page.keyboard.press('Tab');
	const ring = await page.evaluate(() => {
		const el = document.activeElement as HTMLElement | null;
		if (!el || el === document.body) return { focused: false, outlineWidth: '', outlineColor: '' };
		const s = getComputedStyle(el);
		return { focused: true, outlineWidth: s.outlineWidth, outlineColor: s.outlineColor };
	});
	expect(ring.focused, 'Tab must move focus into the app').toBe(true);
	// Assert the app rule specifically, not merely "some outline" — a future UA-default outline would
	// pass a bare `!== 'none'` as a tautology. The rule is `2px solid var(--accent)`, #50d0a0.
	expect(ring.outlineWidth, 'the app :focus-visible ring is 2px').toBe('2px');
	expect(ring.outlineColor, 'the ring colour is --accent (#50d0a0)').toBe('rgb(80, 208, 160)');
});
