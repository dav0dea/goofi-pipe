import { test, expect, type Locator, type Page } from '@playwright/test';
import { closeAddedTab, waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * Put the workspace back after a Split Right. `ws.split` re-arms AppShell's 400ms `set_layout`
 * debounce, which writes into the RUNNING PATCH — one backend for the whole run — so a spec that
 * splits and leaves early persists a 2-panel workspace that every later spec then boots into.
 * It passes alone and depends on nothing but timing, which is why it belongs in a `finally`.
 */
async function closeSplit(page: Page): Promise<void> {
	await page.getByTestId('panel-header').nth(1).getByRole('button', { name: 'Close panel' }).click();
	await expect(page.locator('.panel'), 'the workspace is back to one panel').toHaveCount(1);
	await page.waitForTimeout(700); // past AppShell's 400ms set_layout debounce
}

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

	try {
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
	} finally {
		// The tab half of the rule `closeSplit` states above — this file is the LAST of `default`,
		// so a tab left behind here lands on `touch`, three files and a project away.
		await closeAddedTab(page);
	}
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
	try {
		const choice = page.getByTestId('empty-panel').locator('.choice').first();
		await expect(choice).toBeVisible();
		const choiceFont = await choice.evaluate((el) => getComputedStyle(el).fontFamily);
		expect(choiceFont, 'the empty-panel tile renders in the app mono face').toContain(
			'JetBrains Mono'
		);
	} finally {
		await closeSplit(page);
	}
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

// ── The kept-bespoke button population, characterized (M-Task 7) ────────────────────────────
//
// M-Task 7 strips app.css's base `button` SKIN — background/color/border/border-radius/padding/
// transition plus `:hover`/`:disabled`/`.primary`/`.ghost` — and keeps only the RESET
// (`font: inherit; cursor: pointer`), the same class of rule as `input,select,textarea{font:inherit}`.
// The migrated `<Button>`/`<IconButton>` population provably cannot regress: it styles itself. The
// population that CAN is the deliberately-kept bespoke rows/tiles, which must render from their own
// rule alone. This pins each one's real appearance so a strip that silently hands one back to the
// UA (or drops a radius/transition it was inheriting) fails here instead of in a screenshot nobody
// takes. `--fs-*`/`--space-*` are rem, so the expectations are stated against the live root size.

/** The properties the base skin used to supply, read off one rendered element. */
async function skin(loc: Locator) {
	await expect(loc).toBeVisible();
	return loc.evaluate((el) => {
		const cs = getComputedStyle(el);
		return {
			fontFamily: cs.fontFamily,
			fontSize: parseFloat(cs.fontSize),
			fontWeight: cs.fontWeight,
			background: cs.backgroundColor,
			borderWidth: cs.borderTopWidth,
			radius: cs.borderTopLeftRadius,
			padTop: parseFloat(cs.paddingTop),
			padLeft: parseFloat(cs.paddingLeft),
			transition: cs.transitionProperty,
			rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
		};
	});
}

/** Every kept-bespoke button renders in the app face, never the UA's `400 13.333px Arial`. */
function expectAppFace(s: { fontFamily: string }, who: string) {
	expect(s.fontFamily, `${who} renders in the app mono face`).toContain('JetBrains Mono');
}

test('the kept-bespoke chrome buttons render from their own rules, not the base skin', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const header = page.getByTestId('panel-header').first();
	await header.waitFor();

	// 1. ContextMenu `.item` — a full-bleed menu row: transparent, borderless, --radius-sm so the
	//    accent hover fill has round corners, --space-3/--space-4 pads, its own bg+color transition.
	await header.locator('.content-btn').click();
	const item = await skin(page.locator('.context-menu .item').first());
	expectAppFace(item, 'the context-menu row');
	expect(item.fontSize, 'the row takes the menu --fs-small').toBeCloseTo(0.82 * item.rem, 0);
	expect(item.background, 'the row is transparent at rest').toBe('rgba(0, 0, 0, 0)');
	expect(item.borderWidth, 'the row is borderless').toBe('0px');
	expect(item.radius, 'the row rounds its hover fill (--radius-sm)').toBe('4px');
	expect(item.padTop).toBeCloseTo(0.375 * item.rem, 0);
	expect(item.padLeft).toBeCloseTo(0.5 * item.rem, 0);
	expect(item.transition, 'the row fades its own hover fill').toContain('background');
	await page.keyboard.press('Escape');

	// 2. AddNodeMenu `.item` — the same shape, deliberately square (`border-radius: 0`) because the
	//    rows are full-bleed inside the menu surface. The FIRST row is `.hl`, so probe an unhighlit one.
	// Opened through the automation façade: adding a node is panel-local behaviour, so the app
	// header carries no button for it (topbar.spec.ts pins what it does carry).
	await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
	await page.getByTestId('add-menu-list').waitFor();
	const addItem = await skin(page.locator('.add-menu .item:not(.hl)').first());
	expectAppFace(addItem, 'the add-node row');
	expect(addItem.fontSize).toBeCloseTo(0.82 * addItem.rem, 0);
	expect(addItem.background, 'unhighlighted rows are transparent').toBe('rgba(0, 0, 0, 0)');
	expect(addItem.borderWidth).toBe('0px');
	expect(addItem.radius, 'the add-menu rows are deliberately square').toBe('0px');
	expect(addItem.padTop).toBeCloseTo(0.375 * addItem.rem, 0);
	expect(addItem.padLeft).toBeCloseTo(0.75 * addItem.rem, 0);
	expect(addItem.transition, 'the highlight fades in as the cursor moves').toContain('background');
	await page.keyboard.press('Escape');

	// 3+4. FsBrowser `.root` (sidebar shortcut) and `.entry` (file row) — both list rows whose
	//      hover/selected fill is a rounded accent wash, so the radius is load-bearing here.
	await page.getByTestId('topbar-load').click();
	await page.getByTestId('fs-list').waitFor();
	const root = await skin(page.locator('.roots .root:not(.active)').first());
	expectAppFace(root, 'the fs sidebar root');
	expect(root.fontSize).toBeCloseTo(0.82 * root.rem, 0);
	expect(root.background, 'an inactive root is transparent').toBe('rgba(0, 0, 0, 0)');
	expect(root.borderWidth).toBe('0px');
	expect(root.radius, 'its hover wash is rounded').toBe('4px');
	expect(root.padTop).toBeCloseTo(0.375 * root.rem, 0);
	expect(root.padLeft).toBeCloseTo(0.5 * root.rem, 0);
	expect(root.transition).toContain('background');

	const entry = await skin(page.getByTestId('fs-entry').first());
	expectAppFace(entry, 'the fs file row');
	expect(entry.fontSize).toBeCloseTo(0.82 * entry.rem, 0);
	expect(entry.background, 'an unselected entry is transparent').toBe('rgba(0, 0, 0, 0)');
	expect(entry.borderWidth).toBe('0px');
	expect(entry.radius, 'its hover/selected wash is rounded').toBe('4px');
	expect(entry.padTop).toBeCloseTo(0.25 * entry.rem, 0);
	expect(entry.padLeft).toBeCloseTo(0.75 * entry.rem, 0);
	expect(entry.transition).toContain('background');
	await page.keyboard.press('Escape');

	// 5. EmptyPanel `.choice` — a tile, not a row: the --surface-1 face IS the affordance, the
	//    border is the hover accent alone, and --radius-md rounds the card. Split last, since every
	//    locator above is scoped to `.first()`; the `finally` hands the workspace back either way.
	await header.click({ button: 'right' });
	await page.locator('.context-menu .item', { hasText: 'Split Right' }).click();
	try {
		const choice = await skin(page.getByTestId('empty-panel').locator('.choice').first());
		expectAppFace(choice, 'the empty-panel tile');
		expect(choice.fontSize, 'the tile takes the body size').toBeCloseTo(choice.rem, 0);
		expect(choice.background, 'the tile face is --surface-1').toBe('rgb(28, 28, 28)');
		expect(choice.borderWidth, 'a 1px border it colours only on hover').toBe('1px');
		expect(choice.radius, 'the card is --radius-md').toBe('6px');
		expect(choice.padTop).toBeCloseTo(0.75 * choice.rem, 0);
		expect(choice.padLeft).toBeCloseTo(0.625 * choice.rem, 0);
		expect(choice.transition).toContain('background');
	} finally {
		await closeSplit(page);
	}
});

test('the kept-bespoke node-scoped buttons render from their own rules', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);

	// 6. SlotViewer `.tri` — the disclosure triangle on a node's output slot. Frozen node-canvas
	//    geometry: a 16px chromeless box at the canvas's fixed 10px type size.
	const tri = await skin(page.locator('.slot-viewer .tri').first());
	expectAppFace(tri, 'the slot disclosure triangle');
	expect(tri.fontSize, 'the node canvas is a fixed-px coordinate system').toBe(10);
	expect(tri.background, 'no button chrome at all').toBe('rgba(0, 0, 0, 0)');
	expect(tri.borderWidth).toBe('0px');
	expect(tri.padTop).toBe(0);
	expect(tri.padLeft).toBe(0);

	// 7. ParamForm `.pf-name` — the inspector's click-to-rename title. It must read as the heading
	//    it replaced (600 weight, --fs-strong, no chrome), not as a button.
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	const name = await skin(page.getByTestId('auto-side-panel').getByTestId('node-name'));
	expectAppFace(name, 'the inspector rename title');
	expect(name.fontSize, 'the title is --fs-strong').toBeCloseTo(name.rem, 0);
	expect(name.fontWeight, 'it reads as a heading').toBe('600');
	expect(name.background, 'it carries no button surface').toBe('rgba(0, 0, 0, 0)');
	expect(name.borderWidth).toBe('0px');
	expect(name.padTop).toBe(0);
	expect(name.padLeft).toBe(0);

	await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	await waitForNoNode(page, uid);
});
