import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addGlobal, setGlobalValue, globals } from '../lib/goofi';

/** Borrow the default node-editor panel as a Globals panel, run `body`, then give it back.
 *
 * The layout lives in the RUNNING PATCH and outlives this page (AppShell pushes it on a 400ms
 * debounce), so a spec that leaves without restoring persists a globals-shaped workspace that
 * every later spec boots into. Mirrors `console-rows.spec.ts`. */
async function inGlobalsPanel(page: Page, body: () => Promise<void>): Promise<void> {
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'globals'), panelId);
	try {
		await expect(page.getByTestId('globals-panel')).toBeVisible();
		await body();
	} finally {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		await page.waitForTimeout(700); // settle past the layout debounce before the page goes away
	}
}

test('globals: default_ufreq is seeded, a user global adds, edits round-trip', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	// The system global is always present (mirrored from the manager on sync).
	await expect.poll(async () => (await globals(page)).some((g) => g.name === 'default_ufreq')).toBe(true);
	const seeded = (await globals(page)).find((g) => g.name === 'default_ufreq')!;
	expect(seeded).toMatchObject({ system: true, type: 'float' });

	// Add a user global + edit the system one (command ops — server-validated, resolve void);
	// both round-trip through the doc.
	await addGlobal(page, 'subject', 'P07', 'string');
	await setGlobalValue(page, 'default_ufreq', 45);
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

// A global's VALUE is machine-read: expressions resolve `globals.<name>` and node process()/setup()
// read `ctx.globals`. It must not be autocorrected, autocapitalised or red-underlined as prose. The
// panel's author already declared exactly that one row up, on the NAME field — the `TextInput`
// migration dropped it from the value cell, where the primitive's `text` default supplies the
// opposite (spellcheck on, autocorrect on, sentence capitalisation) and a consumer cannot override
// it by spreading, because the mode attributes are applied AFTER `{...rest}`.
test('a string global’s value cell is typed as machine-read, not prose', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await addGlobal(page, 'm15_token', 'AF7-AF8', 'string');
	try {
		await inGlobalsPanel(page, async () => {
			const cell = page.locator('tr[data-name="m15_token"]').getByTestId('global-value');
			await expect(cell, 'the value round-tripped into the panel').toHaveValue('AF7-AF8');
			await expect(cell).toHaveAttribute('spellcheck', 'false');
			await expect(cell).toHaveAttribute('autocorrect', 'off');
			await expect(cell).toHaveAttribute('autocapitalize', 'off');
			await expect(cell).toHaveAttribute('autocomplete', 'off');
		});
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.removeGlobal('m15_token'));
	}
});

// The panel's own commit path, which `d0f31b1` rebuilt onto the `useLiveValue` latch and which no
// test has crossed since: globals.spec drives values through the `window.goofi` façade, which
// bypasses the widgets entirely. Typing is the part that can break — the latch has to hold the
// buffer while the doc echoes back, and the panel has to parse to the global's DECLARED type
// rather than to whatever the widget handed it.
test('the Globals panel commits a typed number as the global’s declared type', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await addGlobal(page, 'm18_gain', 0, 'float');
	await addGlobal(page, 'm18_count', 0, 'int');
	try {
		await inGlobalsPanel(page, async () => {
			const gain = page.locator('tr[data-name="m18_gain"]').getByTestId('global-value');
			await gain.fill('2.5');
			await gain.blur();
			await expect
				.poll(async () => (await globals(page)).find((g) => g.name === 'm18_gain')?.value)
				.toBeCloseTo(2.5, 5);

			// An int global rounds on commit — the panel parses to the declared type, and 7.6 typed
			// into an int is 8, not a float that silently retypes the global.
			const count = page.locator('tr[data-name="m18_count"]').getByTestId('global-value');
			await count.fill('7.6');
			await count.blur();
			await expect
				.poll(async () => (await globals(page)).find((g) => g.name === 'm18_count')?.value)
				.toBe(8);
		});
	} finally {
		await page.evaluate(() => {
			const c = (window as any).goofi.commands;
			return Promise.all([c.removeGlobal('m18_gain'), c.removeGlobal('m18_count')]);
		});
	}
});
