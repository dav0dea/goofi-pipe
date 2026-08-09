import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

// app.css resets `button { font: inherit }` and `input, select, textarea { font: inherit }` for one
// documented reason: a UA `font` DECLARATION beats inheritance, so those elements would otherwise
// fall out of the app face entirely. `code`/`pre`/`kbd`/`samp` are the same class of element and had
// no such rule — they rendered in the browser's generic monospace, whatever the page around them
// said.
//
// What the reset buys is NOT a face: it is that the face is decided by the cascade rather than by
// the UA. Under the two-face taxonomy that distinction is what these two tests pull apart — the
// same reset hands a CHROME `<code>` the sans it inherits and a DATA `<pre>` the mono its component
// declares, and neither one gets the browser's generic monospace.

// The probe is the one `<code>` in the tree that declares no font of its own: Panel's unknown-type
// fallback, which a `.gfi` layout naming a retired panel reaches for real. It is chrome (an
// explanatory message in a panel body), so with no UA default leaking it renders the body sans.
test('a <code> with no rule of its own inherits the chrome face, not the UA monospace', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	try {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'retired-panel-type'),
			panelId
		);
		const code = page.locator('.panel .missing code');
		await expect(code).toBeVisible();
		const font = await code.evaluate((el) => getComputedStyle(el).fontFamily);
		expect(font, 'the UA reset hands <code> back to the inherited app face').toContain('Inter');
	} finally {
		// Never leave a bogus panel type in the manager's stored layout for the next spec.
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		await page.waitForTimeout(700); // past AppShell's 400ms set_layout debounce
	}
});

// The reset is a RESET, not a skin: `code, pre, kbd, samp` scores (0,0,1), so every component rule
// that sizes one of these elements still wins — the family comes back by inheritance, the RUNG does
// not come from here. The console row is the reachable one, and it is the DATA half of the pair
// above: the mono it renders is the one its own panel declares, arriving through `font: inherit`.
// (The restatements this reset once made look redundant are load-bearing again under two faces —
// a data surface that states no family follows `body` into the chrome face.)
test('the reset hands back family only — a component rule still owns the size', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

	// A Python node whose process() needs a connected ARRAY input raises on every tick; the graph
	// store mirrors each raise into the console as a stderr line. The cheapest real console content.
	const uid = await addNode(page, 'LempelZiv', 'python');
	try {
		await waitForNode(page, uid);
		const txt = page.getByTestId('console-entry').first().locator('.txt');
		await expect(txt, 'the node error reached the console').toBeVisible();

		const s = await txt.evaluate((el) => ({
			family: getComputedStyle(el).fontFamily,
			size: parseFloat(getComputedStyle(el).fontSize),
			rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
		}));
		expect(s.family, 'the <pre> row inherits the app mono face').toContain('JetBrains Mono');
		expect(s.size, 'and keeps its own --fs-small rung, not the UA smaller-monospace').toBeCloseTo(
			0.82 * s.rem,
			0
		);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		await page.waitForTimeout(700);
	}
});
