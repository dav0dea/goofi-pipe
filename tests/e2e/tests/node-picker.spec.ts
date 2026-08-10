import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { BAR_PANELS } from '../lib/panelBar';

/**
 * The second door onto a panel's node binding (the first is a drag from an editor, which needs an
 * editor on the same page and a pointer that can drag between two of them — neither of which a
 * phone, or a layout page holding only viewers, has).
 *
 * One control, `$lib/panels/NodeSelect`, worn identically by every panel type the manager marks
 * `acceptsNode`; these tests drive it through the same testid in each, which is what "in the same
 * way" has to mean to be checkable.
 */

const panelId = (page: Page): Promise<string> =>
	page.evaluate(() => (window as any).goofi.query.panels()[0].panelId);

const panelView = (page: Page, id: string): Promise<{ node: string | null; slot: string | null }> =>
	page.evaluate(
		(pid) => (window as any).goofi.query.panels().find((p: any) => p.panelId === pid),
		id
	);

/** Borrow the sole panel as `type`, and hand it back to the editor afterwards. */
async function borrow(page: Page, type: string, ready: string): Promise<() => Promise<void>> {
	const id = await panelId(page);
	await page.evaluate(
		([p, t]) => (window as any).goofi.commands.setPanelType(p, t),
		[id, type] as const
	);
	await expect(page.getByTestId(ready)).toBeVisible();
	return async () => {
		await page.evaluate(
			(p) => (window as any).goofi.commands.setPanelType(p, 'node-editor'),
			id
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
	};
}

for (const [type, ready] of BAR_PANELS) {
	test(`the ${type} panel binds a node from its bar, with nothing dragged`, async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		const id = await panelId(page);
		const restore = await borrow(page, type, ready);
		try {
			const picker = page.getByTestId('panel-node').locator('select');
			// Present BEFORE anything is bound — an unbound panel that hides its own picker leaves a
			// phone no way in at all, which is the defect this door exists to close.
			await expect(picker, 'the picker is there with nothing bound').toBeVisible();
			expect((await panelView(page, id)).node, 'and nothing is bound yet').toBeNull();

			await picker.selectOption(uid);
			await expect
				.poll(async () => (await panelView(page, id)).node, { timeout: 5_000 })
				.toBe(uid);
			// The option reads as the node's DISPLAY name while the committed value is its uid.
			expect(await picker.locator('option:checked').innerText()).toBe('oscillator0');

			await picker.selectOption('');
			await expect.poll(async () => (await panelView(page, id)).node).toBeNull();
		} finally {
			await restore();
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});
}

test('a binding whose node was removed reads as unbound, never as a raw uid', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	const id = await panelId(page);
	const restore = await borrow(page, 'metadata', 'node-linked-panel');
	try {
		const picker = page.getByTestId('panel-node').locator('select');
		await picker.selectOption(uid);
		await expect.poll(async () => (await panelView(page, id)).node).toBe(uid);

		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);

		await expect
			.poll(async () => picker.inputValue(), { timeout: 5_000 })
			.toBe('');
		expect(
			await picker.locator('option').allInnerTexts(),
			'the dead node is gone from the list too'
		).toEqual(['No node']);
	} finally {
		await restore();
	}
});

/* The viewer binds a node AND a slot, so a pick that only settled half of it would leave the panel
   on a slot name the new node has never had. `linkNodeToPanel` clears the slot in the same merged
   write (pinned exactly, by payload, in `workspace.svelte.test.ts`); what THIS measures is the
   consequence — a viewer picked from the bar is drawing, with no second gesture. */
test('a viewer picked from the bar is fully bound, slot and all', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);
	const id = await panelId(page);
	const restore = await borrow(page, 'viewer', 'node-linked-panel');
	try {
		await page.getByTestId('panel-node').locator('select').selectOption(osc);
		await expect.poll(async () => (await panelView(page, id)).node).toBe(osc);

		const slot = page.getByTestId('viewer-slot').locator('select');
		await expect(slot, 'the slot picker came up with the node').toBeVisible();
		expect(await slot.inputValue(), 'settled on a real output slot, not on nothing').not.toBe('');
		await expect(page.getByTestId('viewer-kind'), 'and the viewer itself is drawing').toBeVisible();
	} finally {
		await restore();
		await page.evaluate((x) => (window as any).goofi.commands.removeNode(x), osc);
		await waitForNoNode(page, osc);
	}
});
