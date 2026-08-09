import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * The loudest line on a node card was internal to it.
 *
 * `GoofiNode`'s `.header` paints a `border-bottom` and `SlotViewer`'s `header` paints a
 * `border-top`, with no margin between them — so the seam between a node's title bar and its first
 * slot rendered at 2 CSS px, against the card's own 1 px outline. On the app's most-repeated
 * element, that is the single biggest contributor to "too salient", and it is not a design choice
 * anyone made: it is two components each drawing their own edge.
 *
 * The tree already contained the intended shape — `PlacementPreview`, documented as mirroring
 * GoofiNode exactly, de-duplicates it with `.viewers .slot-row:first-child { border-top: none }`.
 * This applies that precedent from the child's side, and pins the result in composited pixels
 * rather than in the source, because "how many lines does this seam paint" has no DOM answer.
 *
 * The node header's own `border-bottom` deliberately SURVIVES, and is asserted here: `inputUnits`
 * floors every node body at one unit, so a node with no outputs has no slot row to carry the line
 * and would otherwise lose its header/body separation entirely.
 */
test('the node card draws one line under its header, not two', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	try {
		await waitForNode(page, uid);

		// The card carries no uid of its own; reach it through the slot that does.
		const card = page
			.locator('.goofi-node')
			.filter({ has: page.locator(`.slot-viewer[data-node="${uid}"]`) })
			.first();
		const slotHeader = page.locator(`.slot-viewer[data-node="${uid}"] header`).first();
		await expect(slotHeader, 'the node renders its output slot').toBeVisible();

		expect(
			await slotHeader.evaluate((el) => getComputedStyle(el).borderTopWidth),
			'the first slot row adds no second line under the node header'
		).toBe('0px');

		const nodeHeader = card.locator('.header').first();
		expect(
			await nodeHeader.evaluate((el) => getComputedStyle(el).borderBottomWidth),
			'the separation is still drawn — once — so an output-less node keeps it'
		).toBe('1px');
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});
