import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

test('selection brings the node update rate forward exactly like hover', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [260, 220]);
	try {
		await waitForNode(page, uid);
		const card = page.locator(`.svelte-flow__node[data-id="${uid}"]`);
		const node = card.locator('.goofi-node');
		const rate = node.locator('.rate');
		await expect(rate, 'the first node-stats push supplies the rate').toBeVisible({
			timeout: 15_000
		});

		const opacity = (): Promise<number> =>
			rate.evaluate((el) => parseFloat(getComputedStyle(el).opacity));

		await page.locator('.topbar').hover();
		await expect.poll(opacity, { message: 'unselected and unhovered, the rate rests quietly' }).toBe(0.3);

		await card.hover();
		await expect.poll(opacity, { message: 'hover brings the rate forward' }).toBe(0.85);
		const hovered = await opacity();

		await card.locator('.header').click();
		await expect(node, 'the node accepted the selection').toHaveClass(/selected/);
		await page.locator('.topbar').hover();
		await expect.poll(opacity, { message: 'selection keeps the rate forward after hover leaves' }).toBe(0.85);
		expect(await opacity(), 'selection and hover use the exact same emphasis').toBe(hovered);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});
