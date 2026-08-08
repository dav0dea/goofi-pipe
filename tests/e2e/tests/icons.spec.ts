import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * ONE ICON SYSTEM — the app draws every icon itself, and lets the browser draw none.
 *
 * The defect this pins out was that "an icon" meant two different things depending on where you
 * looked. Most were Unicode glyphs (`✕`, `⋯`, `↶`, `▾`) resolved by the OS font stack, so the app
 * looked different on every platform and had no say in the weight or the shape. The rest were not
 * the app's at all: `MetadataPanel` rendered a bare `<summary>`, whose disclosure marker is drawn
 * by the BROWSER — a triangle in Chrome, a different one in Firefox, in a colour the palette never
 * chose.
 *
 * Both are replaced by one renderer over vendored Lucide geometry (`$lib/ui/Icon`), which is what
 * the `svg[data-icon]` assertions below read. `icons.test.ts` guards the geometry itself; this file
 * guards the thing only the running app can answer — that the browser is no longer drawing.
 *
 * Runs under the `default` (fine-pointer) project; the icons are pointer-independent.
 */

const CHROME_GLYPHS = ['✕', '×', '⋯', '↶', '↷', '▾', '▸', '▶', '＋', '▕', '▁', '⤢', '⟳'];

/** Add an Oscillator and select it, which slides in the inspector — ParamForm + MetadataPanel. */
async function selectedOscillator(page: Page): Promise<string> {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(page.getByTestId('auto-side-panel')).toHaveClass(/open/);
	return uid;
}

async function drop(page: Page, uid: string): Promise<void> {
	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid).catch(() => {});
}

/* The one place the BROWSER was still drawing. `<details>`/`<summary>` is the right element for a
   collapsible meta field and stays — what goes is its UA marker, replaced by the app's own chevron
   so the affordance is the app's shape in the app's ink. */
test('the metadata tree draws its own chevron, and no native <details> marker', async ({ page }) => {
	const uid = await selectedOscillator(page);
	try {
		const field = page.getByTestId('auto-side-panel').locator('.meta-field').first();
		// The panel renders off the live data stream, so the first frame has to land first.
		await expect(field, 'a running Oscillator puts at least one meta field on screen').toBeVisible();
		const summary = field.locator('summary');

		await expect(
			summary,
			'no UA disclosure marker — the browser draws nothing here'
		).toHaveCSS('list-style-type', 'none');
		await expect(
			summary.locator('svg[data-icon]'),
			'the affordance is the app’s own chevron instead'
		).toHaveCount(1);

		// And it is a real affordance, not decoration: it turns with the section it marks.
		await expect(summary.locator('svg[data-icon]')).toHaveAttribute('data-icon', 'chevron-right');
	} finally {
		await drop(page, uid);
	}
});

/* The app header, the tab strip and the panel header are the three resident chrome bars — between
   them they carried nine of the Unicode glyphs. Asserted as a property of the RENDERED text, so a
   glyph that comes back anywhere in those bars fails here rather than at the site that reintroduced
   it. */
test('the resident chrome bars render icons, never a text glyph', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const bars = page.locator('.topbar, [data-testid="panel-header"]');
	await expect(bars, 'the app header and the panel header are both on screen').toHaveCount(2);
	const text = (await bars.allTextContents()).join('');
	for (const g of CHROME_GLYPHS)
		expect(text, `the chrome bars no longer render "${g}" as text`).not.toContain(g);

	// The controls that carried those glyphs each name the icon they draw now.
	for (const [testid, icon] of [
		['topbar-undo', 'undo-2'],
		['topbar-redo', 'redo-2'],
		['topbar-save-caret', 'chevron-down'],
		['topbar-overflow', 'ellipsis'],
		['panel-split-row', 'square-split-horizontal'],
		['panel-split-column', 'square-split-vertical'],
		['panel-maximize', 'maximize-2'],
		['panel-close', 'x']
	] as const) {
		await expect(
			page.getByTestId(testid).locator('svg'),
			`${testid} draws the ${icon} icon`
		).toHaveAttribute('data-icon', icon);
	}
});

/* Phil's call, and the reason it is worth a line of its own: the two splits are Lucide's two
   DRAWN icons, not one icon rotated 90°. A rotation would read as the same control twice, and it
   is what the header's own naming ("a vertical divider ▕ / a horizontal divider ▁") used to say
   in a comment because the glyphs could not say it themselves. */
test('the two panel splits are two distinct icons, neither of them rotated', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const right = page.getByTestId('panel-split-row').locator('svg');
	const down = page.getByTestId('panel-split-column').locator('svg');

	await expect(right).toHaveAttribute('data-icon', 'square-split-horizontal');
	await expect(down).toHaveAttribute('data-icon', 'square-split-vertical');
	await expect(right, 'drawn as-is, not turned').toHaveCSS('transform', 'none');
	await expect(down, 'drawn as-is, not turned').toHaveCSS('transform', 'none');

	// The mapping is unchanged: Split Right still makes a row, Split Down still makes a column.
	await expect(page.getByTestId('panel-split-row')).toHaveAttribute('title', 'Split Right');
	await expect(page.getByTestId('panel-split-column')).toHaveAttribute('title', 'Split Down');
});
