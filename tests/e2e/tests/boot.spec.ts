import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

test('boots the backend and the app connects + loads the catalog', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// The catalog arrived over the control WS: Oscillator + Buffer are always present.
	const types = await page.evaluate(() => (window as any).goofi.query.nodeTypes());
	expect(Array.isArray(types)).toBe(true);
	expect(types.map((t: any) => t.type)).toContain('Oscillator');
});
