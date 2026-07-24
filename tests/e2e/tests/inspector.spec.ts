import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, nodeParams } from '../lib/goofi';

// Characterization e2e for the rebuilt inspector (spec §7, N-Task 5): drive the REAL rendered
// InspectorOverlay — the slide-in ParamForm — against a live Oscillator node, complementing the
// store-level graph.spec.ts. Each case commits through a rendered control and reads the round-trip
// back through the doc (`nodeParams` / `query.node`), so it pins the newly-wired ParamForm, not a
// synthetic fixture. Runs under the `default` (fine-pointer) project.

/** The single-selection slide-in inspector. */
function panel(page: Page) {
	return page.getByTestId('auto-side-panel');
}

/** A node's live name via the read façade. */
function nodeName(page: Page, uid: string): Promise<string | undefined> {
	return page.evaluate((u) => (window as any).goofi.query.node(u)?.name, uid);
}

/** Boot, add an Oscillator, select it so the per-editor inspector slides in, and return its uid. */
async function addAndSelect(page: Page): Promise<string> {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	// Selecting exactly one node opens the editor's inspector overlay (enabled by default).
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(panel(page), 'the inspector slides in for a single selection').toHaveClass(/open/);
	// The backend persists across specs, so the auto-assigned display name is not fixed; assert the
	// header reflects THIS node's actual name (proving the overlay is bound to the selection).
	const name = await nodeName(page, uid);
	await expect(panel(page).getByTestId('node-name')).toHaveText(name!);
	return uid;
}

test.describe('Inspector (real node)', () => {
	test('commits a param through the rendered control and it round-trips through the doc', async ({
		page
	}) => {
		const uid = await addAndSelect(page);
		// amplitude lives in the default (`oscillator`) group — edit its NumberInput and commit on Enter.
		const number = panel(page).getByTestId('param-field-amplitude').getByTestId('param-number');
		await number.fill('0.42');
		await number.press('Enter');
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.value)
			.toBeCloseTo(0.42, 5);
	});

	test('toggles fx on a param and expression_enabled flips', async ({ page }) => {
		const uid = await addAndSelect(page);
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.expression_enabled)
			.toBe(false);
		await panel(page).getByTestId('param-field-amplitude').getByTestId('param-fx-toggle').click();
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.expression_enabled)
			.toBe(true);
	});

	test('inline-renames the node from the header', async ({ page }) => {
		const uid = await addAndSelect(page);
		await panel(page).getByTestId('node-name').click();
		const input = panel(page).getByTestId('node-name-input');
		await input.fill('my_osc');
		await input.press('Enter');
		await expect.poll(() => nodeName(page, uid)).toBe('my_osc');
	});

	test('opens the docs disclosure to reveal the node docstring', async ({ page }) => {
		await addAndSelect(page);
		// The Disclosure keeps its body out of the DOM until opened.
		await expect(panel(page).getByTestId('docstring')).toHaveCount(0);
		await panel(page).getByTestId('docs-toggle').click();
		await expect(panel(page).getByTestId('docstring')).toBeVisible();
	});
});
