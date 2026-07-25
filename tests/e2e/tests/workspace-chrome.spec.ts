import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

// The workspace panel system is FROZEN UX; sub-project M restyled its chrome onto the
// `$lib/ui` primitives (PanelHeader's dropdown + maximize/close, the tab strip's ✕/＋).
// Nothing else in the suite exercises that chrome, so these are the regression guards for
// the two invariants a primitive swap can silently break: the header dropdown's ContextMenu
// wiring, and the tab strip's zero-width collapsed close.

// The active-panel ring must actually be PAINTED, and only a pixel readback can tell: the rule
// was present all along (an `inset box-shadow` in the accent), but an inset shadow paints BELOW
// child content, so on the node editor the header covered its top edge and `.svelte-flow`'s opaque
// background covered the other three. A DOM/class/computed-style assertion passes on that broken
// code — this one samples what the compositor actually produced.
test('an active node-editor panel paints the accent ring at its inner edge', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// The flow canvas is the occluder — probe only once it has painted, or an unpainted body
	// would let the old inset shadow show through and green the probe for the wrong reason.
	await page.locator('.svelte-flow').first().waitFor();

	const panel = page.locator('.panel[data-panel-type="node-editor"]').first();
	await expect(panel, 'the sole default panel is the active one').toHaveClass(/\bactive\b/);
	const box = (await panel.boundingBox())!;

	// Decode the screenshot in-page (the browser owns a PNG decoder; this Node process does not)
	// and scan the 3px band just inside the panel's left edge, below the header and above the
	// flow Controls at bottom-left. --accent (#50d0a0) at 45% over --bg composites to about
	// rgb(45,103,81), so green leads red by ~58; every neutral in that band leads by 0.
	const png = (await page.screenshot()).toString('base64');
	const greenLead = await page.evaluate(
		async ({ png, x, yTop, yBot }) => {
			const img = new Image();
			img.src = `data:image/png;base64,${png}`;
			await img.decode();
			const scale = img.width / window.innerWidth;
			const canvas = document.createElement('canvas');
			canvas.width = img.width;
			canvas.height = img.height;
			const ctx = canvas.getContext('2d')!;
			ctx.drawImage(img, 0, 0);
			const { data } = ctx.getImageData(
				Math.round(x * scale),
				Math.round(yTop * scale),
				Math.max(1, Math.round(3 * scale)),
				Math.round((yBot - yTop) * scale)
			);
			let lead = 0;
			for (let i = 0; i < data.length; i += 4) lead = Math.max(lead, data[i + 1] - data[i]);
			return lead;
		},
		{ png, x: box.x, yTop: box.y + 60, yBot: box.y + box.height / 2 }
	);
	expect(greenLead, 'the accent ring is painted on the panel inner left edge').toBeGreaterThan(20);
});

test('the panel header dropdown opens the context menu and Escape dismisses it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const header = page.getByTestId('panel-header').first();
	await header.waitFor();
	const menu = page.locator('.context-menu');

	await header.locator('.content-btn').click();
	await expect(menu, 'the content dropdown opens a context menu').toHaveCount(1);
	await expect(menu.locator('.item').first()).toBeVisible();

	await page.keyboard.press('Escape');
	await expect(menu, 'Escape dismisses it').toHaveCount(0);
});

test('a collapsed tab close button occupies zero width (the even-padding invariant)', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// A close ✕ only renders once there is more than one tab.
	const tabs = page.getByTestId('workspace-tabs');
	await tabs.getByRole('button', { name: 'New tab' }).click();
	const close = tabs.getByRole('button', { name: 'Close tab' }).first();
	await close.waitFor({ state: 'attached' });

	// Its tab is neither hovered nor active, so the ✕ is collapsed: it must take NO
	// horizontal space at all, or every inactive tab is padded wider than its neighbours.
	// (A primitive with a 1px border clamps `width: 0` to 2px under border-box — the exact
	// regression this guards.)
	await expect
		.poll(async () => (await close.boundingBox())?.width, {
			message: 'the collapsed ✕ takes zero width'
		})
		.toBe(0);

	// Hovering its tab reveals it at the frozen 16px reveal width.
	await close.locator('xpath=..').hover();
	await expect.poll(async () => (await close.boundingBox())?.width).toBe(16);
});

test('the tab strip ＋ keeps its frozen 22px box (not the primitive --hit floor)', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// IconButton floors its box to --hit (28px on a fine pointer). The tab pills are ~23px
	// tall, so an unpinned ＋ would stand visibly taller than the tabs beside it — the tab
	// strip is frozen geometry and pins it back to 22, exactly as `.close` pins 16.
	const add = page.getByTestId('workspace-tabs').getByRole('button', { name: 'New tab' });
	await add.waitFor();
	const box = (await add.boundingBox())!;
	expect(box.width, 'the ＋ keeps its pre-migration 22px width').toBe(22);
	expect(box.height, 'the ＋ keeps its pre-migration 22px height').toBe(22);
});

// The two rows M deliberately kept bespoke (a context-menu item, an empty-panel tile) are
// styled entirely by their own class — including the font. `app.css`'s base `button` rule is
// what M-Task 7 strips, and buttons do NOT inherit font by default, so each must declare
// `font: inherit` itself or fall back to the UA default (Arial 13.333px) the moment it goes.
// The rule is still there today, so the guard SIMULATES its removal: `font: revert` at the same
// (0,0,1) specificity, injected last, hands any button that declares no font back to the UA.
test('the kept-bespoke menu row and panel tile declare their own font', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await page.addStyleTag({ content: 'button { font: revert; }' });

	const header = page.getByTestId('panel-header').first();
	await header.locator('.content-btn').click();
	const item = page.locator('.context-menu .item').first();
	await expect(item).toBeVisible();
	const itemFont = await item.evaluate((el) => getComputedStyle(el).fontFamily);
	expect(itemFont, 'the context-menu row renders in the app mono face').toContain('JetBrains Mono');
	await page.keyboard.press('Escape');

	// A freshly split panel starts empty, which is what renders the choice tiles.
	const menu = page.locator('.context-menu');
	await header.click({ button: 'right' });
	await menu.locator('.item', { hasText: 'Split Right' }).click();
	const choice = page.getByTestId('empty-panel').locator('.choice').first();
	await expect(choice).toBeVisible();
	const choiceFont = await choice.evaluate((el) => getComputedStyle(el).fontFamily);
	expect(choiceFont, 'the empty-panel tile renders in the app mono face').toContain(
		'JetBrains Mono'
	);
});

test('the panel header dropdown keeps its frozen geometry over the primitive padding', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// The header's pin and the primitive's own `.ui-btn.s-md` padding live in separate built CSS
	// chunks, so a specificity tie between them would be settled by the emitted <link> order
	// rather than the source — and the control would silently take --space-6 sides (double).
	const btn = page.getByTestId('panel-header').first().locator('.content-btn');
	await btn.waitFor();
	const box = (await btn.boundingBox())!;
	expect(box.height, 'the header dropdown keeps the 26px bar geometry').toBe(20);

	const { padLeft, rem } = await btn.evaluate((el) => ({
		padLeft: parseFloat(getComputedStyle(el).paddingLeft),
		rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
	}));
	expect(padLeft, 'the header pins --space-3 sides, not the primitive --space-6').toBeCloseTo(
		0.375 * rem,
		0
	);
});
