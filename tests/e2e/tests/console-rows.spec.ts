import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * The Console's virtual scroller keeps its own height model — `estimateH()` in the script mirrors
 * `.row`'s padding (`PAD`) and its text line box (`LINE_H`) in px, because `layout.cum` sums an
 * ESTIMATE for every row the ResizeObserver has not measured yet. When the model and the DOM
 * disagree the thumb reads the wrong length and the content drifts as rows measure in.
 *
 * M's IconButton migration widened that gap: the per-row copy button is rendered unconditionally
 * (only faded out), and the primitive floors its box at `--hit` = 28px on a fine pointer — four
 * times the pre-existing 3px error, and enough to make a single-line row 33px against a 20px
 * estimate. The assertions below are content-length independent: they pin the MODEL (a row is its
 * text box plus the padding and border the estimator accounts for) rather than one row's pixels.
 */
test('a console row is sized by its text, not by its action buttons', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	// The default layout is one node editor; borrow it and give it straight back, so no later spec
	// inherits a console-shaped workspace.
	await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

	// A Python node whose process() needs a connected ARRAY input raises on every tick; the graph
	// store mirrors each raise into the console as a stderr line. The cheapest real console content.
	const uid = await addNode(page, 'LempelZiv', 'python');
	try {
		await waitForNode(page, uid);
		const row = page.getByTestId('console-entry').first();
		await expect(row, 'the node error reached the console').toBeVisible();

		const m = await row.evaluate((el) => {
			const px = (sel: string) =>
				el.querySelector(sel)!.getBoundingClientRect().height;
			const cs = getComputedStyle(el);
			return {
				row: el.getBoundingClientRect().height,
				txt: px('.txt'),
				copy: px('.console-copy-btn'),
				pad: parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom),
				border: parseFloat(cs.borderBottomWidth) + parseFloat(cs.borderTopWidth)
			};
		});

		expect(m.copy, 'the copy button fits inside the row line box it shares').toBeLessThanOrEqual(
			m.txt
		);
		expect(m.row, 'the row is exactly text + padding + border — what estimateH models').toBeCloseTo(
			m.txt + m.pad + m.border,
			0
		);
		// And the estimator's own constants are the ones the DOM actually uses.
		expect(m.pad, 'PAD mirrors the row padding').toBe(4);
		expect(m.border, 'the estimate accounts for the row border').toBe(1);
		expect(m.txt % 16, 'the text box is a whole number of LINE_H lines').toBe(0);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
	}
});
