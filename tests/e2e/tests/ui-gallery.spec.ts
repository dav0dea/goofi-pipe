import { test, expect } from '@playwright/test';

// The /dev/ui primitive gallery is a static, backend-free showcase, so we do NOT wait on
// window.goofi (the AppShell/control-WS readiness gate the graph specs use) — we wait on the
// rendered samples. This spec is the "failing test first" for <Button>/<IconButton>: it fails
// before the route + primitives exist. Runs under the `default` (fine-pointer) project.
test.describe('UI primitives gallery', () => {
	test('renders every Button and IconButton variant/size sample', async ({ page }) => {
		await page.goto('/dev/ui');
		// A spread across variants + sizes for both primitives.
		await expect(page.getByTestId('ui-button-default-md')).toBeVisible();
		await expect(page.getByTestId('ui-button-primary-md')).toBeVisible();
		await expect(page.getByTestId('ui-button-ghost-sm')).toBeVisible();
		await expect(page.getByTestId('ui-button-danger-md')).toBeVisible();
		await expect(page.getByTestId('ui-button-disabled')).toBeDisabled();
		await expect(page.getByTestId('ui-icon-primary-md')).toBeVisible();
		await expect(page.getByTestId('ui-icon-danger-sm')).toBeVisible();
	});

	test('a keyboard-focused Button shows the app accent focus ring', async ({ page }) => {
		await page.goto('/dev/ui');
		await page.getByTestId('ui-button-default-sm').waitFor();
		// Keyboard Tab (not programmatic focus) so :focus-visible engages; the first focusable
		// element on this static page is the first sample button.
		await page.keyboard.press('Tab');
		const ring = await page.evaluate(() => {
			const el = document.activeElement as HTMLElement | null;
			if (!el || el === document.body) return { tag: '', testid: '', outlineWidth: '', outlineColor: '' };
			const s = getComputedStyle(el);
			return {
				tag: el.tagName,
				testid: el.getAttribute('data-testid') ?? '',
				outlineWidth: s.outlineWidth,
				outlineColor: s.outlineColor
			};
		});
		expect(ring.tag, 'Tab moves focus onto a gallery button').toBe('BUTTON');
		expect(ring.testid, 'the focused element is a UI primitive sample').toMatch(/^ui-button-/);
		// Assert the app rule specifically (2px solid --accent = #50d0a0), not merely "some outline",
		// so a future UA-default outline could not pass this as a tautology.
		expect(ring.outlineWidth, 'the app :focus-visible ring is 2px').toBe('2px');
		expect(ring.outlineColor, 'the ring colour is --accent (#50d0a0)').toBe('rgb(80, 208, 160)');
	});
});
