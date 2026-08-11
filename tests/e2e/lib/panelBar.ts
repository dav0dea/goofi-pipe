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

/** The controls a bound panel's toolbar draws, left to right. Only the two panels that add their
 *  own controls next to the node picker are listed: parameters and console hold the picker alone,
 *  so they have no gap of their own to get wrong. */
const BAR_CONTROLS: ReadonlyArray<[type: string, selectors: string[]]> = [
	[
		'viewer',
		[
			'.ui-status-dot',
			'[data-testid="panel-node"]',
			'[data-testid="viewer-slot"]',
			'[data-testid="viewer-kind"]',
			'[data-testid="viewer-settings-cog"]'
		]
	],
	['metadata', ['.ui-status-dot', '[data-testid="panel-node"]', '[data-testid="metadata-slot"]']]
];

/**
 * One strip, one gap. A panel toolbar's controls are one row of siblings to the eye, so the space
 * between any two of them is the strip's own `--bar-gap` — not a wrapper's private gap, and not
 * that plus a margin. The viewer bar read as "node picker … slot·kind" because the panel-specific
 * controls sat in a nested div carrying `margin-left` AND a narrower gap of its own: 17.1px then
 * 5.1px, against the bar's 6.8px.
 *
 * Measured off the rendered boxes rather than the CSS, because that is what the eye reads: the
 * defect was invisible to every rule in isolation and visible the moment the rects were compared.
 */
export async function controlsSitAtOneGap(page: Page): Promise<void> {
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	try {
		for (const [type, selectors] of BAR_CONTROLS) {
			await page.evaluate(
				([id, t, u]) => {
					(window as any).goofi.commands.setPanelType(id, t);
					(window as any).goofi.commands.bindNodeToPanel(id, u);
				},
				[panelId, type, uid] as const
			);
			await expect(page.getByTestId('node-linked-panel')).toBeVisible();
			await expect(page.locator(selectors[selectors.length - 1])).toBeVisible();
			const m = await page.evaluate((sels) => {
				const bar = document.querySelector('.panel-body .ui-bar .ui-bar-group')!;
				const boxes = sels.map((s) => {
					const el = bar.closest('.ui-bar')!.querySelector(s);
					return el ? el.getBoundingClientRect() : null;
				});
				return {
					gap: parseFloat(getComputedStyle(bar).columnGap),
					gaps: boxes.map((b, i) => (i === 0 || !b || !boxes[i - 1] ? null : b.left - boxes[i - 1]!.right))
				};
			}, selectors);
			for (let i = 1; i < selectors.length; i++) {
				expect(
					m.gaps[i],
					`the ${type} bar spaces ${selectors[i]} from ${selectors[i - 1]} by the strip's own gap`
				).toBeCloseTo(m.gap, 1);
			}
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
