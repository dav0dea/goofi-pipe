/**
 * Undo as a BUTTON reaches it. That undo/redo work at all — that a stack walks back to an empty
 * patch and forward to the same uids — is the Rust suite's (`editing.rs`), driven through the one
 * op vocabulary with no browser in the way. What can only be asked here is whether a control in
 * the app records a step at all.
 */

import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode, undo, redo } from '../lib/goofi';

/** The bound node of the sole panel, as the façade reports it. */
function boundNode(page: import('@playwright/test').Page): Promise<string | null> {
	return page.evaluate(() => (window as any).goofi.query.panels()[0].node);
}

/**
 * The unlink ✕ is an EDIT — it dirties the patch — but it used to write the panel state straight
 * into the layout tree with no history entry of its own. Layout undo restores a whole
 * `WorkspaceState` snapshot captured at the last tracked action, so an unrecorded write landing
 * after one is in neither `before` nor `after`: an unrelated Ctrl+Z destroyed it and the redo did
 * not bring it back. The store op is what the vitest pins; this pins the BUTTON reaching it.
 */
test('the unlink ✕ earns its own undo step, which restores the binding', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	try {
		await page.evaluate(
			([id, u]) => {
				(window as any).goofi.commands.setPanelType(id, 'parameters');
				(window as any).goofi.commands.bindNodeToPanel(id, u);
			},
			[panelId, uid] as const
		);
		// A panel write is a COMMAND now, so the binding appears when the manager's delta does —
		// polled, like every other assertion in this test.
		await expect.poll(() => boundNode(page), { message: 'the panel is bound' }).toBe(uid);

		await page.getByTestId('node-linked-panel').getByRole('button', { name: 'Unlink node' }).click();
		await expect.poll(() => boundNode(page), { message: 'the ✕ unbinds' }).toBe(null);
		expect(
			await page.evaluate(() => (window as any).goofi.query.undoLabel()),
			'the click named its own undo step'
		).toBe('Unbind node from panel');

		await undo(page);
		await expect.poll(() => boundNode(page), { message: 'undo re-binds' }).toBe(uid);
		await redo(page);
		await expect.poll(() => boundNode(page), { message: 'redo unbinds again' }).toBe(null);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
	}
});
