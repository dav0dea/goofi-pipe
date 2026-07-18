import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addGlobal, setGlobalValue, globals } from '../lib/goofi';

test('globals: default_ufreq is seeded, a user global adds, edits round-trip', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	// The system global is always present (mirrored from the manager on sync).
	await expect.poll(async () => (await globals(page)).some((g) => g.name === 'default_ufreq')).toBe(true);
	const seeded = (await globals(page)).find((g) => g.name === 'default_ufreq')!;
	expect(seeded).toMatchObject({ system: true, type: 'float' });

	// Add a user global + edit the system one; both round-trip through the doc.
	expect(await addGlobal(page, 'subject', 'P07', 'string')).toBe(true);
	expect(await setGlobalValue(page, 'default_ufreq', 45)).toBe(true);
	await expect
		.poll(async () => (await globals(page)).find((g) => g.name === 'subject')?.value)
		.toBe('P07');
	await expect
		.poll(async () => (await globals(page)).find((g) => g.name === 'default_ufreq')?.value)
		.toBeCloseTo(45, 5);
});

test('the Globals panel renders when opened', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await page.evaluate(() => (window as any).goofi.commands.addTab('globals'));
	await expect(page.getByTestId('globals-panel')).toBeVisible();
});
