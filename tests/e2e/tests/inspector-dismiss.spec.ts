import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

/**
 * The selection inspector had no way out (D-R9, carryover **C9**).
 *
 * It slides in over its editor whenever exactly one node is selected. Its only off-switch — the
 * editor's `inspector-toggle` corner control — sits at `z-index: 5` UNDER the pane's 50, so the
 * moment the pane opens the control that closes it is buried; and the other exit, tapping empty
 * canvas to deselect, needs canvas the pane covers. On a phone that was a dead end. On desktop it
 * was merely obscure, which is the same defect (D-R2: one system, one fix).
 *
 * A `default`-project spec on purpose: desktop is the reference, and the fix is not coarse-gated.
 */

const pane = (page: Page) => page.getByTestId('auto-side-panel');

async function addAndSelect(page: Page): Promise<string> {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(pane(page), 'a single selection opens the inspector').toHaveClass(/open/);
	return uid;
}

test.afterEach(async ({ page }) => {
	await page.evaluate(() => {
		const g = (window as any).goofi;
		const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
		if (uids.length) return g.commands.removeNodes(uids);
	});
});

test('the inspector has a dismiss control, and it closes the pane', async ({ page }) => {
	await addAndSelect(page);
	const close = pane(page).getByTestId('inspector-close');
	await expect(close, 'the pane carries its own way out').toBeVisible();
	await close.click();
	// Turning the inspector off unmounts the pane entirely (`{#if enabled}`) — it does not merely
	// slide out, so the canvas underneath is genuinely handed back.
	await expect(pane(page), 'and it closes').toHaveCount(0);
});

test('the buried toggle is not left under the pane, and it brings the pane back', async ({
	page
}) => {
	const uid = await addAndSelect(page);
	// While the pane is open the toggle is covered at every width (the pane is pinned to the right
	// edge and the toggle sits 10px inside it), so leaving it in the tree only means an invisible,
	// tabbable control under an opaque surface.
	await expect(page.getByTestId('inspector-toggle')).toHaveCount(0);

	await pane(page).getByTestId('inspector-close').click();
	const toggle = page.getByTestId('inspector-toggle');
	await expect(toggle, 'closing hands the affordance back').toBeVisible();
	await expect(toggle).toHaveAttribute('aria-pressed', 'false');

	// …and it is a toggle, not a one-way door: pressing it re-arms the pane for the same selection.
	await toggle.click();
	await expect(pane(page), 'the same node is inspected again').toHaveClass(/open/);
	await expect(pane(page).getByTestId('node-name')).toHaveText(
		(await page.evaluate((u) => (window as any).goofi.query.node(u).name, uid)) as string
	);
});

test('Escape closes the inspector by clearing the selection', async ({ page }) => {
	await addAndSelect(page);
	// The editor's own keydown handler answers Escape when focus is not in a field; clearing the
	// selection is what closes the pane. Pinned because it is the keyboard exit D-R9 says is
	// missing — it is not, and a future guard must not quietly take it away.
	await page.locator('.svelte-flow__pane').first().focus().catch(() => {});
	await page.keyboard.press('Escape');
	await expect(pane(page)).not.toHaveClass(/open/);
});
