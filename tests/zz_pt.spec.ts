import { test } from '@playwright/test';
import { waitForApp } from '../lib/app';

test('panel type menu', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const header = page.getByTestId('panel-header').first();
	await header.click({ button: 'right' });
	const items = await page.locator('.context-menu .item').allTextContents();
	console.log('MENU:', JSON.stringify(items));
	await page.keyboard.press('Escape');
	const pid = await page.evaluate(() => (window as any).goofi.query.panels()[0]?.panelId);
	console.log('PANEL ID:', pid);
});
