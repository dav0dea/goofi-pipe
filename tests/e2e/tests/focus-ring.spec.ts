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

// ...and shows it exactly ONCE. app.css's `input:focus { border-color: var(--accent) }` (0,0,1) does
// not beat `:focus-visible` (0,1,0), so a focused field used to paint the accent twice: a 1px accent
// border, a 1px gap, then the 2px accent outline. Against the "too salient" brief that fires the
// loudest ink in the palette — the same ink as node selection and the active-panel ring — twice for
// one state. Only a computed readback can tell the two rings apart.
test('a focused text field paints ONE accent ring, not two concentric ones', async ({ page }) => {
	await page.goto('/dev/ui');
	const input = page.getByTestId('ui-field-number');
	await input.waitFor();
	await input.click();

	const ring = await input.evaluate((el) => {
		const s = getComputedStyle(el);
		return {
			focused: document.activeElement === el,
			outlineWidth: s.outlineWidth,
			outlineColor: s.outlineColor,
			borderColor: s.borderTopColor
		};
	});
	expect(ring.focused, 'clicking the field focuses it').toBe(true);
	// Browsers match :focus-visible on a text input even under mouse focus (a UA heuristic), so the
	// outline half is the real, rendered ring — not a proxy for it.
	expect(ring.outlineWidth, 'the single ring is the 2px :focus-visible outline').toBe('2px');
	expect(ring.outlineColor, 'and it is --accent').toBe('rgb(80, 208, 160)');
	expect(ring.borderColor, 'the field keeps its resting --border hairline (no second ring)').toBe(
		'rgb(72, 72, 72)'
	);
});
