import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * `control.ts` states the split outright: the uid is the universal identity everything references
 * the node by, and the name is "for the label only". Two labels broke the second half by rendering
 * the identity — the add-node menu's seed chip (`from 000000000003.out`, on the primary
 * add-and-auto-wire flow) and the console's per-row source button, which sat beside canvas nodes
 * named `oscillator0`. Both are the one place their surface names the node at all, and neither was
 * pinned, so a resolution that regresses would ship silently.
 *
 * Asserted as "shows the name and NOT the 12-hex uid", because the uid legitimately stays in the
 * DOM around both (`data-node`, the click's focus/autoLink routing) — only the visible text moved.
 */
const HEX_UID = /^[0-9a-f]{12}$/;

test('the add-menu seed chip names the node it will wire from', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const uid = await addNode(page, 'Oscillator', 'inputs', [60, 60]);
	await waitForNode(page, uid);
	expect(uid, 'the identity really is an opaque hex string').toMatch(HEX_UID);
	const name: string = await page.evaluate((u) => (window as any).goofi.query.node(u).name, uid);

	try {
		// The connector pill click — the flow that opens the menu seeded from a slot.
		await page
			.locator(`.svelte-flow__node[data-id="${uid}"]`)
			.getByTestId('slot-output-pin')
			.first()
			.click();
		const chip = page.getByTestId('add-menu-seed');
		await expect(chip, 'the menu opened seeded from that slot').toBeVisible();
		await expect(chip.locator('.seed-ref')).toHaveText(`${name}.out`);
	} finally {
		await page.keyboard.press('Escape');
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});

test('a console row names its source node', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	// Borrow the default editor panel and give it straight back (see console-rows.spec.ts).
	await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

	// A Python node whose process() needs a connected ARRAY input raises every tick — the cheapest
	// real console content.
	const uid = await addNode(page, 'LempelZiv', 'python');
	try {
		await waitForNode(page, uid);
		const name: string = await page.evaluate((u) => (window as any).goofi.query.node(u).name, uid);
		expect(name, 'the display name is not the identity').not.toMatch(HEX_UID);

		const source = page.getByTestId('console-entry').first().locator('button.node');
		await expect(source, 'the row attributes the raise to a node').toBeVisible();
		await expect(source).toHaveText(name);
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
