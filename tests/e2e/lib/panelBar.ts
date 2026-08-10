import { expect, type Page } from '@playwright/test';
import { addNode, waitForNode, waitForNoNode } from './goofi';

/** Every panel type whose body opens with a content toolbar (`$lib/ui/Bar`), and the testid that
 *  says the new content has actually drawn — a panel retype swaps the whole subtree, so measuring
 *  before that lands measures the OUTGOING panel's bar. */
export const BAR_PANELS: ReadonlyArray<[type: string, ready: string]> = [
	['parameters', 'node-linked-panel'],
	['viewer', 'node-linked-panel'],
	['metadata', 'node-linked-panel'],
	['console', 'console-panel']
];

/**
 * The user's rule, measured: a panel's toolbar is exactly as tall as the panel header above it, in
 * every panel that has one. Both come from the same token (`--panel-header-h`: 26px on a fine
 * pointer, 44px under a coarse one), so this is an equality and not a bound — a toolbar that grew
 * by its own padding, or by one control that did not take the strip's dense box, fails it by the
 * exact number of pixels it grew.
 *
 * Driven with a node BOUND, which is each panel's tallest configuration: the status dot, the slot
 * and viewer-kind dropdowns and the unlink ✕ are all only there once something is bound.
 */
export async function barsMatchTheHeader(page: Page): Promise<void> {
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	try {
		for (const [type, ready] of BAR_PANELS) {
			await page.evaluate(
				([id, t, u]) => {
					(window as any).goofi.commands.setPanelType(id, t);
					(window as any).goofi.commands.bindNodeToPanel(id, u);
				},
				[panelId, type, uid] as const
			);
			await expect(page.getByTestId(ready)).toBeVisible();
			const m = await page.locator('.panel').first().evaluate((el) => ({
				header: el.querySelector('.panel-header')!.getBoundingClientRect().height,
				bar: el.querySelector('.panel-body .ui-bar')!.getBoundingClientRect().height
			}));
			expect(
				m.bar,
				`the ${type} panel's toolbar is the panel header's own height (${m.header})`
			).toBeCloseTo(m.header, 1);
		}
	} finally {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
}
