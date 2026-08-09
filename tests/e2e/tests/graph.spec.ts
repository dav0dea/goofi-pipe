import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, nodes, waitForNode, waitForNoNode, nodeParams, updateParam } from '../lib/goofi';

test('adds a node through the façade and it appears in the graph', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	try {
		expect(uid).toBeTruthy();
		// The node appears only after the doc round-trip — wait for it, then assert its identity.
		await waitForNode(page, uid);
		const present = await nodes(page);
		expect(present.find((n) => n.uid === uid)!.type).toBe('Oscillator');
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});

test('edits a param and the value round-trips through the doc', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	try {
		// The leaf-write no-ops until the node is in the client's doc replica — wait first.
		await waitForNode(page, uid);
		await updateParam(page, uid, 'oscillator', 'amplitude', 0.42);
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.value)
			.toBeCloseTo(0.42, 5);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});
