import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * Who owns a key press — the editor panel, the browser, or the modal on top of it.
 *
 * `NodeEditorPanel`'s window-level `onKeydown` guards on two things only: is my panel the active
 * one, and is the target a text field. Neither question is the one that matters at this boundary,
 * which is why it was carried over from the M audit (R2-3) — and then never entered a task brief:
 *
 *  - **Tab** was `preventDefault`ed for EVERY non-field target, with no `shiftKey` check. The first
 *    Tab looks fine because it opens the add-node menu and the menu focuses its search — but
 *    `{#if menuOpen}` is unkeyed, so re-entering `openAddMenu` with the menu already open neither
 *    remounts it nor re-fires that focus. Every later Tab is a bare `preventDefault` with nothing
 *    to refocus: a one-way trap (WCAG 2.1.2), and the reason no chrome outside the canvas — the tab
 *    strip, the header actions, the inspector's ✕, R's own `.conn-label` focus reveal — was ever
 *    Tab-reachable.
 */

/** A short, stable name for whatever currently has focus. */
const activeName = (page: Page): Promise<string | null> =>
	page.evaluate(() => {
		const el = document.activeElement as HTMLElement | null;
		if (!el) return null;
		return el.dataset.testid ?? el.tagName.toLowerCase();
	});

test('Tab on a chrome control is the browser’s, not the canvas’s', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// The node editor is the active panel from the first frame (DEFAULT_PANEL_TYPE + `_focusFirst`),
	// and no keyboard event ever clears that — which is why this reproduces in the boot layout.
	await page.getByTestId('topbar-load').focus();
	await page.keyboard.press('Tab');

	await expect(
		page.getByTestId('add-node-menu-anchor'),
		'the canvas does not claim a Tab it was not given'
	).toHaveCount(0);
	expect(await activeName(page), 'focus moved on to the next control in the header').toBe(
		'topbar-overflow'
	);
});

test('focus can leave the add-node menu again', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
	const menu = page.getByTestId('add-node-menu-anchor');
	await expect(menu).toBeVisible();
	await expect(page.getByTestId('add-menu-search'), 'the menu takes focus on open').toBeFocused();

	const outside = (): Promise<boolean> =>
		page.evaluate(() => {
			const a = document.activeElement;
			const m = document.querySelector('[data-testid="add-node-menu-anchor"]');
			return !!a && !!m && !m.contains(a);
		});
	let left = false;
	// Bounded: the menu holds a search field plus one row per palette entry, so a few dozen presses
	// is generous. Without the fix this never leaves — the second Tab lands on a row <button> and
	// every press after it is a bare preventDefault.
	for (let i = 0; i < 60 && !left; i++) {
		await page.keyboard.press('Tab');
		left = await outside();
	}
	expect(left, 'Tab is not a one-way door into the palette (WCAG 2.1.2)').toBe(true);

	await page.keyboard.press('Escape');
	await expect(menu).toHaveCount(0);
});

test('Tab on the bare canvas still opens the add-node menu', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// The Blender-style shortcut is the point of the branch and must survive its scoping: with
	// nothing focused, the canvas is what the key press is for.
	await page.keyboard.press('Tab');
	const menu = page.getByTestId('add-node-menu-anchor');
	await expect(menu).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(menu).toHaveCount(0);
});
