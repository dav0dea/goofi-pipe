import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, nodes, waitForNode, waitForNoNode, undo, redo, canUndo } from '../lib/goofi';

// Regression contract: add-node → undo removes it → redo restores it. Written against the
// CURRENT custom history; must keep passing across the Phase 4/5 Y.UndoManager migration.
test('undo removes an added node and redo restores it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const before = (await nodes(page)).length;

	const uid = await addNode(page, 'Buffer', 'signal');
	await waitForNode(page, uid);
	await expect.poll(() => canUndo(page)).toBe(true);

	await undo(page);
	await waitForNoNode(page, uid);
	expect((await nodes(page)).length).toBe(before);

	await redo(page);
	await waitForNode(page, uid);
});
