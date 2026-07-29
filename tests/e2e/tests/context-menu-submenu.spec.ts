import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * The desktop reference for `workspace/ContextMenu.svelte`'s submenus, pinned because R-Task 7
 * moved the hover listener from `mouseenter` to `pointerenter` (so the pointer type is knowable)
 * and gave a parent row a click action it did not have.
 *
 * Both halves matter here: hovering must still expand — that is the fine-pointer behaviour and it
 * is the reference — and clicking a parent must OPEN rather than toggle, or the click that follows
 * a hover would close the very submenu the hover just opened.
 */

function openHeaderMenu(page: Page) {
	return page.getByTestId('panel-header').first().click({ button: 'right' });
}

const parentRow = (page: Page) =>
	page.locator('.context-menu .item').filter({ hasText: 'Change content' }).first();

test('hovering a parent row still expands its submenu', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await openHeaderMenu(page);
	await expect(page.locator('.context-menu')).toHaveCount(1);

	await parentRow(page).hover();
	await expect(page.locator('.context-menu'), 'the submenu opened on hover').toHaveCount(2);
	await page.keyboard.press('Escape');
	await expect(page.locator('.context-menu')).toHaveCount(0);
});

test('clicking a parent row opens its submenu and never closes it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await openHeaderMenu(page);

	// The hover that precedes any real click has already opened it; the click must leave it open.
	await parentRow(page).click();
	await expect(page.locator('.context-menu')).toHaveCount(2);
	await parentRow(page).click();
	await expect(page.locator('.context-menu'), 'a second click is not a toggle').toHaveCount(2);
	await page.keyboard.press('Escape');
	await expect(page.locator('.context-menu')).toHaveCount(0);
});
