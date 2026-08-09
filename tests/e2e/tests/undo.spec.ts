import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, nodes, waitForNode, waitForNoNode, undo, redo, canUndo } from '../lib/goofi';

/** The bound node of the sole panel, as the façade reports it. */
function boundNode(page: import('@playwright/test').Page): Promise<string | null> {
	return page.evaluate(() => (window as any).goofi.query.panels()[0].node);
}

// Regression contract: add-node → undo removes it → redo restores it (manager-owned history).
test('undo removes an added node and redo restores it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const before = (await nodes(page)).length;

	const uid = await addNode(page, 'Buffer', 'signal');
	try {
		await waitForNode(page, uid);
		await expect.poll(() => canUndo(page)).toBe(true);

		await undo(page);
		await waitForNoNode(page, uid);
		expect((await nodes(page)).length).toBe(before);

		// Redo restores the SAME uid (the history entry is uid-stable), so the finally can remove it.
		await redo(page);
		await waitForNode(page, uid);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});

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
		expect(await boundNode(page)).toBe(uid);

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
		await page.waitForTimeout(700); // past AppShell's 400ms set_layout debounce
	}
});
