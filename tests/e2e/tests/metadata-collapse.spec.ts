import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode, updateParam } from '../lib/goofi';

/**
 * THE METADATA TREE OPENS CLOSED, AND ITS HEADER LINE IS A HINT — not the value.
 *
 * `MetadataPanel` used to expand every small field on the first frame, so a node with a handful of
 * meta keys drew a wall of text before the user asked for any of it. Phil's call: collapsed by
 * default, in BOTH places the component is mounted — the editor's slide-in inspector and the
 * dockable Metadata panel, which are the same component behind `showHeader`.
 *
 * The second test is the one that earns its keep. The collapse choice has to survive the next
 * frame, and this panel re-renders at the data rate: the `open` state used to be re-asserted from
 * Svelte state on every frame while the `toggle` event that REPORTS the user's click fires
 * asynchronously, so a frame landing in that gap silently undid the click. The fix is that the
 * `<details>` now owns its own open state (collapsed is its own default), which is why this test
 * provokes a frame it can see rather than waiting on one it hopes for.
 *
 * The third pins the header/body split that only the running app can show: both come from the same
 * meta value, and the two-decimal cap sits on the header alone.
 *
 * Runs under the `default` (fine-pointer) project; nothing here is pointer-dependent.
 */

/** Add an Oscillator and select it, which slides in the inspector — ParamForm + MetadataPanel. */
async function selectedOscillator(page: Page): Promise<string> {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(page.getByTestId('auto-side-panel')).toHaveClass(/open/);
	return uid;
}

async function drop(page: Page, uid: string): Promise<void> {
	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid).catch(() => {});
}

/** The `sfreq` field, addressed by the key it shows rather than by position — the Oscillator grows
 * a `ufreq` field a second in, which shifts every index after the first. */
function sfreqField(page: Page, root = page.locator('body')) {
	return root.locator('.meta-field', { has: page.locator('.mk', { hasText: /^sfreq$/ }) });
}

test('every metadata field starts collapsed, in the inspector and in the Metadata panel', async ({
	page
}) => {
	const uid = await selectedOscillator(page);
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	try {
		const inspector = page.getByTestId('auto-side-panel');
		// The panel renders off the live data stream, so the first frame has to land first.
		await expect(
			inspector.locator('.meta-field').first(),
			'a running Oscillator puts at least one meta field on screen'
		).toBeVisible();
		await expect(
			inspector.locator('.meta-field[open]'),
			'the inspector draws no field expanded'
		).toHaveCount(0);

		// The dockable panel is the same component with `showHeader = false`; it takes its node from
		// the panel binding rather than the selection, and must answer the same.
		await page.evaluate(
			([id, u]) => {
				const g = (window as any).goofi.commands;
				g.setPanelType(id, 'metadata');
				g.bindNodeToPanel(id, u);
			},
			[panelId, uid] as const
		);
		await expect(
			page.getByTestId('metadata-slot'),
			'the Metadata panel is showing the bound node'
		).toBeVisible();
		await expect(page.locator('.meta-field').first()).toBeVisible();
		await expect(
			page.locator('.meta-field[open]'),
			'and neither does the Metadata panel'
		).toHaveCount(0);
	} finally {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		await drop(page, uid);
		// AppShell pushes the layout into the RUNNING PATCH on a 400ms debounce, and the patch
		// outlives this page — settle past it so no later spec boots into a metadata-shaped workspace.
		await page.waitForTimeout(700);
	}
});

test('a field the user opens stays open when the next frame lands', async ({ page }) => {
	const uid = await selectedOscillator(page);
	try {
		const field = sfreqField(page, page.getByTestId('auto-side-panel'));
		await expect(field, 'the Oscillator tags every frame with its sample rate').toBeVisible();
		await expect(field.locator('.mp')).toHaveText('250');
		await expect(field, 'it starts collapsed, like every field').not.toHaveAttribute('open');

		await field.locator('summary').click();
		await expect(field, 'the click opened it').toHaveAttribute('open', '');

		// Provoke a frame whose meta demonstrably differs, rather than waiting on one that might
		// carry the same text — a re-render the assertion cannot see would make this vacuous.
		await updateParam(page, uid, 'oscillator', 'sfreq', 400);
		await expect(field.locator('.mp'), 'a new frame has rendered').toHaveText('400');
		await expect(field, 'and the user’s choice outlived it').toHaveAttribute('open', '');
	} finally {
		await drop(page, uid);
	}
});

test('the header line caps a scalar at two decimals; the expanded body keeps it whole', async ({
	page
}) => {
	const uid = await selectedOscillator(page);
	try {
		const field = sfreqField(page, page.getByTestId('auto-side-panel'));
		await expect(field).toBeVisible();

		await updateParam(page, uid, 'oscillator', 'sfreq', 333.333333);
		await expect(field.locator('.mp'), 'the header line rounds to two places').toHaveText(
			'333.33'
		);

		await field.locator('summary').click();
		await expect(field.locator('.mv'), 'the body is where the real value lives').toHaveText(
			'333.333333'
		);
	} finally {
		await drop(page, uid);
	}
});
