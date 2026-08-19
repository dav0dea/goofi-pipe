// The panel workspace: its chrome, how a panel splits and reflows, and how the arrangement
// converges between tabs.

import { test, expect, type Locator, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { closeAddedTab, waitForApp, closeSplit, splitRight } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { installInkProbe } from '../lib/ink';
import { barsMatchTheHeader, controlsSitAtOneGap } from '../lib/panelBar';

test.describe('the panel chrome', () => {
	/**
	 * Put the workspace back after a Split Right. `ws.split` is a command against the running patch,
	 * debounce, which writes into the RUNNING PATCH — one backend per worker — so a spec that
	 * splits and leaves early persists a 2-panel workspace that every later spec there then boots into.
	 * It passes alone and depends on nothing but timing, which is why it belongs in a `finally`.
	 */
	async function closeSplit(page: Page): Promise<void> {
		await page.getByTestId('panel-header').nth(1).getByRole('button', { name: 'Close panel' }).click();
		await expect(page.locator('.panel'), 'the workspace is back to one panel').toHaveCount(1);
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

	test('a tab keeps one width — the ✕ holds its seat and only its INK comes and goes', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		// A close ✕ only renders once there is more than one tab. The RESTED state under test is
		// the INACTIVE tab's (an active tab shows its ✕ by design), with the mouse parked away —
		// the ＋ click leaves the pointer where a tab now sits, which would hover-reveal it.
		const tabs = page.getByTestId('workspace-tabs');
		await tabs.getByRole('button', { name: 'New tab' }).click();
		const tab = tabs.locator('.ui-tab:not(.active)').first();
		const close = tab.getByRole('button', { name: 'Close tab' });
		await close.waitFor({ state: 'attached' });
		await page.mouse.move(400, 300);

		try {
			// Phil (2026-08-08): the ✕'s width is RESERVED on every closable tab, hovered or not —
			// a tab that grows on hover jumps its neighbours sideways. At rest the ✕ is merely
			// invisible (opacity) and inert (pointer-events), never absent: an invisible button
			// that still took taps would close an inactive tab a hybrid touchscreen meant to
			// select.
			const rested = (await tab.boundingBox())!.width;
			expect((await close.boundingBox())!.width, 'the rested ✕ keeps its 16px seat').toBe(16);
			await expect
				.poll(() => close.evaluate((el) => getComputedStyle(el).opacity), {
					message: 'the rested ✕ is invisible'
				})
				.toBe('0');
			expect(
				await close.evaluate((el) => getComputedStyle(el).pointerEvents),
				'the invisible ✕ takes no taps'
			).toBe('none');

			// Hovering reveals the ink — and moves NOTHING.
			await tab.hover();
			await expect
				.poll(() => close.evaluate((el) => getComputedStyle(el).opacity), {
					message: 'hover reveals the ✕'
				})
				.toBe('1');
			expect((await tab.boundingBox())!.width, 'the tab width does not jump on hover').toBe(
				rested
			);
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

	test('the layout tab’s label INK sits on the header midline, level with the ＋', async ({
		page
	}) => {
		await installInkProbe(page);
		await page.goto('/');
		await waitForApp(page);

		// Phil (2026-08-08, twice): first the tab's text centre-aligns with the ＋ so the strip
		// reads as one row — then the sharper cut: the ✕ and the row are RIGHT, the label's INK
		// is what rides high. A line box reserves ascent and descent the glyphs need not use, so
		// centring the BOX is not centring what you see. Nothing compensates for that any more —
		// the --ink-nudge token and the three transforms that spent it were deleted with the
		// two-face flip, and this label renders in the chrome sans — which leaves the geometry to
		// be right on its own and makes the MEASUREMENT the point: `lib/ink.ts`'s shared probe
		// (canvas TextMetrics, extent derived at a reference size) against the ＋'s centre, the
		// row's reference, never the label's line box.
		const tabs = page.getByTestId('workspace-tabs');
		const add = tabs.getByRole('button', { name: 'New tab' });
		const ab = (await add.boundingBox())!;
		const d = await tabs
			.locator('.ui-tab-label')
			.first()
			.evaluate((label) => window.__inkMetrics(label));
		expect(
			Math.abs(d.fontBoxCheck),
			'canvas font metrics agree with the layout font box'
		).toBeLessThanOrEqual(1);
		expect(
			Math.abs(d.inkCenter - (ab.y + ab.height / 2)),
			'the tab label’s ink and the ＋ share a vertical centre'
		).toBeLessThanOrEqual(0.8);
	});

	// The two rows M deliberately kept bespoke (a context-menu item, an empty-panel tile) are
	// styled entirely by their own class — including the font. `app.css`'s base `button` rule is
	// what M-Task 7 strips, and buttons do NOT inherit font by default, so each must declare
	// `font: inherit` itself or fall back to the UA default (Arial 13.333px) the moment it goes.
	// The rule is still there today, so the guard SIMULATES its removal: `font: revert` at the same
	// (0,0,1) specificity, injected last, hands any button that declares no font back to the UA.
	// Both rows are CHROME (a menu row, a panel-choice tile), so the face they must come back holding
	// is the sans body face — what this asserts is that the app's own rule decides it, not the UA.
	test('the kept-bespoke menu row and panel tile declare their own font', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await page.addStyleTag({ content: 'button { font: revert; }' });

		const header = page.getByTestId('panel-header').first();
		await header.locator('.content-btn').click();
		const item = page.locator('.context-menu .item').first();
		await expect(item).toBeVisible();
		const itemFont = await item.evaluate((el) => getComputedStyle(el).fontFamily);
		expect(itemFont, 'the context-menu row renders in the app chrome face').toContain('Inter');
		await page.keyboard.press('Escape');

		// A freshly split panel starts empty, which is what renders the choice tiles.
		const menu = page.locator('.context-menu');
		await header.click({ button: 'right' });
		await menu.locator('.item', { hasText: 'Split Right' }).click();
		try {
			const choice = page.getByTestId('empty-panel').locator('.choice').first();
			await expect(choice).toBeVisible();
			const choiceFont = await choice.evaluate((el) => getComputedStyle(el).fontFamily);
			expect(choiceFont, 'the empty-panel tile renders in the app chrome face').toContain('Inter');
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

	/**
	 * Every kept-bespoke button renders in an APP face, never the UA's `400 13.333px Arial` — and in
	 * the one its surface is classified as (D-T3), since this population straddles the taxonomy: menu
	 * rows and tiles are chrome, a file path and a node name are data. Passing the side in is what
	 * keeps a bespoke rule that quietly changes face a red here rather than a screenshot nobody takes.
	 */
	function expectAppFace(s: { fontFamily: string }, who: string, face: 'sans' | 'mono') {
		expect(s.fontFamily, `${who} renders in the app ${face} face`).toContain(
			face === 'mono' ? 'JetBrains Mono' : 'Inter'
		);
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
		expectAppFace(item, 'the context-menu row', 'sans');
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
		expectAppFace(addItem, 'the add-node row', 'sans');
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
		//      Opened in SAVE mode: the same modal, plus the filename field the load footer does not
		//      carry — so both of this dialog's text fields are reachable from one open. (Save reaches
		//      the modal at all because the patch is unnamed, which `waitForApp` above asserts.)
		await page.getByTestId('topbar-save').click();
		await page.getByTestId('fs-list').waitFor();
		const root = await skin(page.locator('.roots .root:not(.active)').first());
		expectAppFace(root, 'the fs sidebar root', 'sans');
		expect(root.fontSize).toBeCloseTo(0.82 * root.rem, 0);
		expect(root.background, 'an inactive root is transparent').toBe('rgba(0, 0, 0, 0)');
		expect(root.borderWidth).toBe('0px');
		expect(root.radius, 'its hover wash is rounded').toBe('4px');
		expect(root.padTop).toBeCloseTo(0.375 * root.rem, 0);
		expect(root.padLeft).toBeCloseTo(0.5 * root.rem, 0);
		expect(root.transition).toContain('background');

		const entry = await skin(page.getByTestId('fs-entry').first());
		expectAppFace(entry, 'the fs file row', 'mono');
		expect(entry.fontSize).toBeCloseTo(0.82 * entry.rem, 0);
		expect(entry.background, 'an unselected entry is transparent').toBe('rgba(0, 0, 0, 0)');
		expect(entry.borderWidth).toBe('0px');
		expect(entry.radius, 'its hover/selected wash is rounded').toBe('4px');
		expect(entry.padTop).toBeCloseTo(0.25 * entry.rem, 0);
		expect(entry.padLeft).toBeCloseTo(0.75 * entry.rem, 0);
		expect(entry.transition).toContain('background');

		// The dialog's two TEXT FIELDS, on the same taxonomy as the rows above them (D-T3): a filesystem
		// path and a patch's name are data, not chrome. `TextInput` declares no family by design (it is
		// `font: inherit` all the way down), so the face can only come from the strip that encloses it —
		// which is exactly the seam a restyle can drop without changing a line of the component.
		expectAppFace(await skin(page.getByTestId('fs-path-input')), 'the fs path bar', 'mono');
		expectAppFace(await skin(page.getByTestId('fs-filename')), 'the fs filename field', 'mono');
		await page.keyboard.press('Escape');

		// 5. EmptyPanel `.choice` — a tile, not a row: the --surface-1 face IS the affordance, the
		//    border is the hover accent alone, and --radius-md rounds the card. Split last, since every
		//    locator above is scoped to `.first()`; the `finally` hands the workspace back either way.
		await header.click({ button: 'right' });
		await page.locator('.context-menu .item', { hasText: 'Split Right' }).click();
		try {
			const choice = await skin(page.getByTestId('empty-panel').locator('.choice').first());
			expectAppFace(choice, 'the empty-panel tile', 'sans');
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
		expectAppFace(tri, 'the slot disclosure triangle', 'mono');
		expect(tri.fontSize, 'the node canvas is a fixed-px coordinate system').toBe(10);
		expect(tri.background, 'no button chrome at all').toBe('rgba(0, 0, 0, 0)');
		expect(tri.borderWidth).toBe('0px');
		expect(tri.padTop).toBe(0);
		expect(tri.padLeft).toBe(0);

		// 7. ParamForm `.pf-name` — the inspector's click-to-rename title. It must read as the heading
		//    it replaced (600 weight, --fs-strong, no chrome), not as a button.
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		const name = await skin(page.getByTestId('auto-side-panel').getByTestId('node-name'));
		expectAppFace(name, 'the inspector rename title', 'mono');
		expect(name.fontSize, 'the title is --fs-strong').toBeCloseTo(name.rem, 0);
		expect(name.fontWeight, 'it reads as a heading').toBe('600');
		expect(name.background, 'it carries no button surface').toBe('rgba(0, 0, 0, 0)');
		expect(name.borderWidth).toBe('0px');
		expect(name.padTop).toBe(0);
		expect(name.padLeft).toBe(0);

		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	});

	/* One dropdown, worn by every panel bar. The viewer-TYPE picker used to hardcode the compact
	   toolbar face while every other dropdown took app.css's form-row box, so two controls in one bar
	   were two different heights. The face is the `Select` primitive's `density="chrome"` now; this is
	   what makes "they match" a rule rather than a coincidence between two files. */
	test('the dropdowns in one panel bar wear one box', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, uid);
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		try {
			await page.evaluate(
				([id, u]) => {
					(window as any).goofi.commands.setPanelType(id, 'viewer');
					(window as any).goofi.commands.bindNodeToPanel(id, u);
				},
				[panelId, uid] as const
			);
			const slot = page.getByTestId('viewer-slot').locator('select');
			const kind = page.getByTestId('viewer-kind').locator('select');
			await expect(kind).toBeVisible();
			const [a, b] = [(await slot.boundingBox())!, (await kind.boundingBox())!];
			expect(a.height, 'the slot picker takes the viewer-type dropdown’s box').toBeCloseTo(
				b.height,
				1
			);
			// …and that box is a STRIP's: shorter than the bar it sits in, which the form-row box is not.
			const bar = (await page.getByTestId('node-linked-panel').locator('.ui-bar').boundingBox())!;
			expect(a.height, 'a bar dropdown is not the tallest thing in the bar').toBeLessThan(bar.height);
		} finally {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});
});

test.describe('the panel bar', () => {
	/* One height for every chrome strip. A panel's toolbar was its content's height plus its own
	   vertical padding — 34.8px under a 26px header on a desktop, 53px under a 44px one on a phone —
	   so the two strips a panel stacks read as two unrelated bands. `touch-panel-bar.spec.ts` is the
	   same assertion under a coarse pointer, where the floor and the height pull hardest against each
	   other and the bar has NO padding left to spend. */
	test('every panel’s toolbar is exactly as tall as the panel header above it', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await barsMatchTheHeader(page);
	});

	/* And one gap across that strip. The height rule above says the toolbar is one band; this says its
	   contents read as one row — the node picker and the panel's own controls are siblings, not two
	   clusters separated by a margin. */
	test('a panel’s toolbar spaces every control it holds by the strip’s own gap', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await controlsSitAtOneGap(page);
	});
});

test.describe('splitting a panel from its corner', () => {
	/**
	 * The corner grip, under a mouse — the half `touch-panel-split.spec.ts` takes away.
	 *
	 * Blender-style: drag a panel corner inward and the panel splits along the dominant drag axis.
	 * `gesture-cancel.spec.ts` covers what a CANCELLED corner drag must not leave behind, but nothing
	 * covered the gesture actually working, so "hide the grips on touch" had no fine-pointer guard to
	 * fail if it took the desktop gesture with it. This is that guard: the grip rests invisible, comes
	 * up on hover, hit-tests, and a completed drag really restructures the workspace.
	 */

	/** The point 3px inside the panel body's top-right corner — inside the grip's clipped triangle. */
	async function topRight(page: Page): Promise<{ x: number; y: number }> {
		const body = (await page.locator('.panel-body').first().boundingBox())!;
		return { x: Math.round(body.x + body.width - 3), y: Math.round(body.y + 3) };
	}

	test('a panel corner hit-tests as a grip and comes up on hover', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const at = await topRight(page);
		const hit = await page.evaluate(
			(p) => (document.elementFromPoint(p.x, p.y) as HTMLElement | null)?.className ?? '',
			at
		);
		expect(hit, 'the fine-pointer grip is the topmost box in the corner').toContain('corner');

		// It rests at opacity 0 and the body's hover brings the set of them up. Retrying, because the
		// reveal is a --dur-slow transition rather than a class flip.
		const grip = page.locator('.panel-body .corner.tr').first();
		await expect(grip).toHaveCSS('opacity', '0');
		await page.locator('.panel-body').first().hover();
		await expect(grip).not.toHaveCSS('opacity', '0');
	});

	test('dragging a corner inward splits the panel', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panels = page.locator('.panel');
		const at = await topRight(page);

		await page.mouse.move(at.x, at.y);
		await page.mouse.down();
		// Leftward, dominantly horizontal: a row split whose new panel takes the right-hand share.
		await page.mouse.move(at.x - 120, at.y + 8);
		await page.mouse.move(at.x - 240, at.y + 10);
		await expect(page.locator('.drag-ghost'), 'the split is previewed while dragging').toHaveCount(1);

		try {
			await page.mouse.up();
			await expect(panels, 'the release commits the split').toHaveCount(2);
			await expect(page.locator('.drag-ghost'), 'and the preview is dropped').toHaveCount(0);
		} finally {
			await closeSplit(page);
		}
	});
});

test.describe('a panel header that does not fit', () => {
	/**
	 * The panel header's progressive overflow — D-R6's arithmetic (`editor/overflowFit.ts`) at its
	 * second consumer.
	 *
	 * A panel header is not the app header: its width is the PANEL's, not the window's, so two panels
	 * side by side on a laptop are already narrower than the app header ever gets on a phone. That is
	 * why this is a `default`-project spec and not a touch one — the collapse keys on available width,
	 * and this is the cheapest place to drive width across its whole range.
	 *
	 * Two things separate it from `topbar-overflow.spec.ts`:
	 *   · the ✕ is NOT in the plan. It is the one control that must be reachable at every width, so it
	 *     never spills, and the header must never push it out of its own panel either.
	 *   · the ⋯ is NOT resident. Its menu holds the spilled actions and nothing else, so at a width
	 *     where all three fit there is no menu to open and the trigger is not drawn.
	 */

	/** The header's overflow-able actions, in DOM order (which is also the order they are given up). */
	const ACTIONS = ['panel-split-row', 'panel-split-column', 'panel-maximize'];

	const hdr = (page: Page, n = 0): Locator => page.getByTestId('panel-header').nth(n);

	/** Which of the three actions the header is currently RENDERING, in DOM order (a spilled one stays
	 *  in the tree at `display: none`, so its width can be re-read when the root size moves). */
	function inHeader(page: Page, n = 0): Promise<string[]> {
		return hdr(page, n).evaluate(
			(el, ids) =>
				[...el.querySelectorAll<HTMLElement>('button[data-testid]')]
					.filter((b) => ids.includes(b.dataset.testid!) && b.offsetParent !== null)
					.map((b) => b.dataset.testid!),
			ACTIONS
		);
	}

	/** Resize and let the ResizeObserver settle (it runs after layout, before the next paint). */
	async function widthTo(page: Page, width: number): Promise<void> {
		await page.setViewportSize({ width, height: 720 });
		await page.evaluate(
			() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
		);
	}

	function menuRow(page: Page, label: string): Locator {
		return page
			.locator('.context-menu .item')
			.filter({ has: page.locator('.label', { hasText: new RegExp(`^${label}$`) }) });
	}

	async function openOverflow(page: Page, n = 0): Promise<void> {
		await hdr(page, n).getByTestId('panel-overflow').click();
		await expect(page.locator('.context-menu').first()).toBeVisible();
	}

	const panels = (page: Page): Locator => page.locator('.panel');

	/** Narrow the window until the first panel's header has given up all three actions, and answer the
	 *  width that did it. Searched rather than hardcoded: how narrow "narrow" is depends on the panel
	 *  type's own name, on the number of panels sharing the window, and on wherever the responsive
	 *  root-size clamp lands — none of which is what these tests are about. */
	async function collapseFully(page: Page): Promise<number> {
		for (const w of [900, 760, 640, 560, 480, 420, 360, 320]) {
			await widthTo(page, w);
			if ((await inHeader(page)).length === 0) return w;
		}
		return 0;
	}

	/** Hand a workspace of any size back as one panel.
	 *  `closeSplit` asserts its way down to one in a single step, so it cannot unwind three. */
	async function restoreSinglePanel(page: Page): Promise<void> {
		while ((await panels(page).count()) > 1) {
			await hdr(page, 1).getByRole('button', { name: 'Close panel' }).click();
			await page.waitForTimeout(100);
		}
		await expect(panels(page), 'the workspace is back to one panel').toHaveCount(1);
	}

	test('a wide panel keeps all three actions in its header, with no ⋯ beside them', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 1400);

		expect(await inHeader(page), 'every action is in the header').toEqual(ACTIONS);
		await expect(
			hdr(page).getByTestId('panel-overflow'),
			'a trigger onto an empty menu is a door into an empty room'
		).toBeHidden();
	});

	test('a narrow panel gives its actions up, lowest priority first', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 1400);
		await splitRight(page); // two panels: each header is half the window wide

		try {
			// The precondition, stated: a half of a 1400px window is still wide enough for all three.
			// Without it the walk below is satisfiable by a header that never had them.
			expect(await inHeader(page), 'a 700px panel keeps every action').toEqual(ACTIONS);

			const left: string[] = [];
			let prev = ACTIONS;
			for (let w = 1400; w >= 320; w -= 20) {
				await widthTo(page, w);
				const now = await inHeader(page);
				for (const id of now)
					expect(prev.includes(id), `${id} came BACK into the header at ${w}px`).toBe(true);
				for (const id of prev) if (!now.includes(id)) left.push(id);
				prev = now;
			}
			expect(left, 'the header gives its actions up in the declared priority order').toEqual(ACTIONS);
		} finally {
			await widthTo(page, 1280);
			await closeSplit(page);
		}
	});

	test('the ✕ is in the header at every width, and never outside its own panel', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 1400);
		await splitRight(page);

		try {
			for (const w of [1400, 1000, 800, 640, 520, 440, 380, 320]) {
				await widthTo(page, w);
				const close = hdr(page).getByRole('button', { name: 'Close panel' });
				await expect(close, `the ✕ is rendered at ${w}px`).toBeVisible();

				// Visible is not enough: the panel clips its overflow, so a header whose line runs long
				// pushes the ✕ out of the box the user can see while it keeps a bounding rect.
				const box = (await close.boundingBox())!;
				const panel = (await panels(page).first().boundingBox())!;
				expect(box.x, `the ✕ starts inside its panel at ${w}px`).toBeGreaterThanOrEqual(panel.x - 1);
				expect(box.x + box.width, `and ends inside it at ${w}px`).toBeLessThanOrEqual(
					panel.x + panel.width + 1
				);
			}
		} finally {
			await widthTo(page, 1280);
			await closeSplit(page);
		}
	});

	test('the header actions act: Split Right, Split Down and Maximize', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 1400);
		expect(await inHeader(page), 'all three are inline at this width').toEqual(ACTIONS);

		// Maximize — `maximizedPanelId` lives outside `WorkspaceState`, so this provably cannot reach
		// the arrangement or the `.gfi`, and it is read back through the button's own label flip.
		await hdr(page).getByRole('button', { name: 'Maximize panel' }).click();
		await expect(hdr(page).getByRole('button', { name: 'Restore panel' })).toBeVisible();
		await hdr(page).getByRole('button', { name: 'Restore panel' }).click();
		await expect(hdr(page).getByRole('button', { name: 'Maximize panel' })).toBeVisible();

		// Split Right — a row split, so the new panel lands beside this one.
		await hdr(page).getByTestId('panel-split-row').click();
		try {
			await expect(panels(page), 'Split Right added a panel').toHaveCount(2);
			const [a, b] = await panels(page).evaluateAll((els) =>
				els.map((e) => e.getBoundingClientRect())
			);
			expect(b.left, 'and it sits beside the original, not under it').toBeGreaterThan(a.left);
		} finally {
			await closeSplit(page);
		}

		// Split Down — a column split: the new panel lands under this one.
		await hdr(page).getByTestId('panel-split-column').click();
		try {
			await expect(panels(page), 'Split Down added a panel').toHaveCount(2);
			const [a, b] = await panels(page).evaluateAll((els) =>
				els.map((e) => e.getBoundingClientRect())
			);
			expect(b.top, 'and it sits under the original, not beside it').toBeGreaterThan(a.top);
		} finally {
			await closeSplit(page);
		}
	});

	test('a spilled action is reachable as a row in the ⋯ menu — and only then', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 1400);
		await splitRight(page);

		/** Each action's row label in the overflow menu — the SAME wording the right-click structural
		 *  menu uses, because the two are one command with two representations (D-R2). */
		const AS_ROW: Record<string, string> = {
			'panel-split-row': 'Split Right',
			'panel-split-column': 'Split Down',
			'panel-maximize': 'Maximize'
		};

		try {
			// Narrow until the header has given every action up, so all three rows are in the menu.
			expect(
				await collapseFully(page),
				'a panel narrow enough to give up all three exists in range'
			).toBeGreaterThan(0);

			await openOverflow(page);
			for (const id of ACTIONS) await expect(menuRow(page, AS_ROW[id]), AS_ROW[id]).toBeVisible();
			await page.keyboard.press('Escape');
			await expect(page.locator('.context-menu')).toHaveCount(0);

			// …and a row that is still a button in the header must NOT also be a row: two doors onto one
			// action is how the two representations drift apart.
			await widthTo(page, 1400);
			expect(await inHeader(page), 'the header took its actions back').toEqual(ACTIONS);
			await expect(hdr(page).getByTestId('panel-overflow')).toBeHidden();
		} finally {
			await widthTo(page, 1280);
			await closeSplit(page);
		}
	});

	test('a menu row acts: Split Down from the ⋯ menu splits the panel', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await splitRight(page);

		try {
			expect(
				await collapseFully(page),
				'a width at which every action is behind the ⋯ exists in range'
			).toBeGreaterThan(0);

			await openOverflow(page);
			await menuRow(page, 'Split Down').click();
			await expect(panels(page), 'the row really split the panel').toHaveCount(3);
			await expect(page.locator('.context-menu')).toHaveCount(0);
		} finally {
			await widthTo(page, 1280);
			await restoreSinglePanel(page);
		}
	});
});

test.describe('the panel surface ladder', () => {
	/**
	 * The workspace's own written rule (app.css: "surface steps carry separation so 1px lines
	 * disappear") applied at the level that repeats most.
	 *
	 * `.panel` used to paint `background: var(--bg)` — byte-identical to the ground behind it — plus a
	 * `1px solid var(--border)` frame, and `.panel-header` painted `--surface-1` with a second hairline
	 * 26px inside that frame. So a panel was distinguishable from the workspace ONLY by lines, and every
	 * split seam stacked three of them across the splitter's 8px span: panelA's border, the splitter's
	 * rule, panelB's border.
	 *
	 * These are computed-result guards, not class assertions: the fill is read off the element and the
	 * seam is counted in real composited pixels, because "how many lines does this seam paint" has no
	 * DOM answer.
	 */

	test('a panel is a surface on the ground, not a rectangle drawn on it', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const surfaces = await page.locator('.panel').first().evaluate((el) => {
			const header = el.querySelector('.panel-header')!;
			const cs = getComputedStyle(el);
			return {
				panel: cs.backgroundColor,
				panelBorder: cs.borderTopColor,
				ground: getComputedStyle(document.body).backgroundColor,
				header: getComputedStyle(header).backgroundColor,
				headerBorder: getComputedStyle(header).borderBottomWidth
			};
		});

		expect(surfaces.panel, 'the panel fill is a real surface step off the ground').not.toBe(
			surfaces.ground
		);
		// The 1px box stays (it is the inset the active ring lives in — see Panel.svelte) but it paints
		// nothing: the surface step is what separates a panel from the ground, not a frame.
		expect(surfaces.panelBorder, 'and it draws no frame, because the step separates').toBe(
			'rgba(0, 0, 0, 0)'
		);
		expect(surfaces.header, 'the header is one further step, not the same surface').not.toBe(
			surfaces.panel
		);
		expect(surfaces.headerBorder, 'so it needs no hairline either').toBe('0px');
	});

	test('a split seam paints one neutral hairline, not a stack of them', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await splitRight(page);
		try {
			await seamRunsAreOne(page);
		} finally {
			await closeSplit(page);
		}
	});

	/** The pixel readback, factored out so the split above can be undone in a `finally`. */
	async function seamRunsAreOne(page: Page): Promise<void> {
		const splitter = page.locator('.splitter.row').first();
		await splitter.waitFor();
		const seam = (await splitter.boundingBox())!;

		// Scan one composited pixel row straight through the seam, 12px on either side of the splitter,
		// at the vertical middle of the panels (clear of both headers and of the flow's corner controls).
		// --border (#484848 → 72) reads far brighter than either fill it can sit on (--bg #111111 → 17,
		// --surface-1 #1c1c1c → 28), and the splitter's centered 1px rule lands on a half pixel so it
		// composites to ≈44. 35 therefore sits above every fill and below every line, antialiased or not.
		const png = (await page.screenshot()).toString('base64');
		const runs = await page.evaluate(
			async ({ png, x0, x1, y }) => {
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
					Math.round(x0 * scale),
					Math.round(y * scale),
					Math.round((x1 - x0) * scale),
					1
				);
				// Count RUNS of bright NEUTRAL columns, so an antialiased line still counts once and the
				// newly-active panel's accent ring — which is state, not chrome — is not counted as one.
				// A --border grey is r≈g≈b; --ring-accent over any fill here leads green by ~58.
				let n = 0;
				let inRun = false;
				for (let i = 0; i < data.length; i += 4) {
					const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
					const line = Math.max(r, g, b) >= 35 && g - r < 20;
					if (line && !inRun) n++;
					inRun = line;
				}
				return n;
			},
			{ png, x0: seam.x - 12, x1: seam.x + seam.width + 12, y: seam.y + seam.height / 2 }
		);

		// Exactly one: the splitter's own rule. (Before, this scan read the two panel borders as well —
		// three lines across an 8px span, the third of them tinted by the active panel's ring.)
		expect(runs, 'the seam reads through the splitter rule alone').toBe(1);
	}

	/** The mean channel value the compositor actually produced over a CSS-px rect, 0–255. */
	async function meanPixel(
		page: Page,
		rect: { x: number; y: number; width: number; height: number }
	): Promise<number> {
		const png = (await page.screenshot()).toString('base64');
		return page.evaluate(
			async ({ png, r }) => {
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
					Math.round(r.x * scale),
					Math.round(r.y * scale),
					Math.max(1, Math.round(r.width * scale)),
					Math.max(1, Math.round(r.height * scale))
				);
				let sum = 0;
				for (let i = 0; i < data.length; i += 4) sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
				return sum / (data.length / 4);
			},
			{ png, r: rect }
		);
	}

	/** The id of the sole default panel. */
	async function solePanelId(page: Page): Promise<string> {
		const panels = await page.evaluate(() => (window as any).goofi.query.panels());
		return panels[0].panelId;
	}

	/** Hand the panel back to the node editor, past the layout debounce (see `closeSplit`). */
	async function restoreEditor(page: Page, panelId: string): Promise<void> {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
	}

	/** The largest (green − red) the compositor produced over a CSS-px rect. --accent (#50d0a0) at
	 * 45% over any neutral in the chrome leads green by ~58; every neutral leads by 0. */
	async function greenLead(
		page: Page,
		rect: { x: number; y: number; width: number; height: number }
	): Promise<number> {
		const png = (await page.screenshot()).toString('base64');
		return page.evaluate(
			async ({ png, r }) => {
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
					Math.round(r.x * scale),
					Math.round(r.y * scale),
					Math.max(1, Math.round(r.width * scale)),
					Math.max(1, Math.round(r.height * scale))
				);
				let lead = 0;
				for (let i = 0; i < data.length; i += 4) lead = Math.max(lead, data[i + 1] - data[i]);
				return lead;
			},
			{ png, r: rect }
		);
	}

	/**
	 * The active-panel ring is ONE mechanism, owned by `Panel`, and it frames the WHOLE panel —
	 * header included — whatever the content happens to be.
	 *
	 * Four panel types used to opt out of it (a `contentOutline` flag on the registry entry) and draw
	 * their own ring around just their body instead, so on Viewer/Parameters/Metadata/Globals the panel
	 * header sat OUTSIDE the focus indication while on the Node Editor/Console/Empty it sat inside it —
	 * the same state drawn two different shapes depending on which panel had focus. Only a pixel
	 * readback can tell them apart: both spellings put an `active` class on a live rule.
	 */
	const PANEL_TYPES = ['node-editor', 'console', 'viewer', 'parameters', 'metadata', 'globals', 'empty'];

	test('every panel type rings the whole panel, its header included', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId = await solePanelId(page);
		const panel = page.locator('.panel').first();
		try {
			for (const type of PANEL_TYPES) {
				await page.evaluate(
					([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
					[panelId, type] as const
				);
				await expect(panel).toHaveAttribute('data-panel-type', type);
				await expect(panel, 'the sole panel is the active one').toHaveClass(/\bactive\b/);
				const box = (await panel.boundingBox())!;
				const headerH = await panel
					.locator('.panel-header')
					.evaluate((el) => el.getBoundingClientRect().height);

				// Beside the header: a 3px band down the panel's left edge, spanning the header's height.
				const beside = await greenLead(page, {
					x: box.x,
					y: box.y + 2,
					width: 3,
					height: headerH - 4
				});
				expect(beside, `${type}: the ring runs down the panel edge beside its header`).toBeGreaterThan(
					20
				);

				// And above it: the panel's own top edge, sampled mid-width clear of the corner radius.
				const above = await greenLead(page, {
					x: box.x + box.width / 2 - 20,
					y: box.y,
					width: 40,
					height: 2
				});
				expect(above, `${type}: the ring closes above its header`).toBeGreaterThan(20);
			}
		} finally {
			await restoreEditor(page, panelId);
		}
	});

	/**
	 * The boundary the first test above is blind to. M-10 deleted `.panel-header`'s hairline AND
	 * promoted `Bar` to the header's own `--surface-2`, so on the four panel types that render a `Bar`
	 * flush at the top of the panel body the two strips merged into one undelimited ~92px chrome slab:
	 * no line, and no step either. D5 says a surface step carries the separation *instead of* a border —
	 * deleting the border with nothing behind it is not a saliency win, it is a lost boundary.
	 *
	 * Asserted on the RESULT, not on which token supplies it: whatever the ladder does, these two
	 * adjacent strips must differ, and neither may buy that back with a hairline.
	 */
	test('a content toolbar flush under the panel header is a step off it, not one slab', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId = await solePanelId(page);
		try {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'console'),
				panelId
			);
			await expect(page.getByTestId('console-panel')).toBeVisible();

			const m = await page.locator('.panel').first().evaluate((el) => {
				const cs = (n: Element) => getComputedStyle(n);
				const header = el.querySelector('.panel-header')!;
				const body = el.querySelector('.panel-body')!;
				const bar = el.querySelector('.panel-body .ui-bar')!;
				return {
					header: cs(header).backgroundColor,
					headerBorder: cs(header).borderBottomWidth,
					bar: cs(bar).backgroundColor,
					barBorder: cs(bar).borderTopWidth,
					flush: bar.getBoundingClientRect().top - body.getBoundingClientRect().top
				};
			});

			expect(m.flush, 'the toolbar really does sit flush under the header').toBeLessThanOrEqual(1);
			expect(m.bar, 'so the two strips must be a real surface step apart').not.toBe(m.header);
			expect(m.headerBorder, 'and the step is what separates them, not a restored hairline').toBe(
				'0px'
			);
			expect(m.barBorder, 'from either side').toBe('0px');
		} finally {
			await restoreEditor(page, panelId);
		}
	});

	/**
	 * The same promotion's other half. `IconButton`'s ghost hover fill was the ABSOLUTE `--surface-2` —
	 * which is exactly the strip a ghost in chrome sits on once `Bar` and `.panel-header` took that
	 * rung — so hovering one painted nothing at all. A ghost control has no surface of its own, so its
	 * hover has to LIFT whatever it happens to sit on; naming a rung can only ever be right on one.
	 *
	 * The reachable, uncompensated instance is `NodeLinkedPanel`'s unlink ✕ (the panel header's own
	 * buttons at least brighten their ink). Only a pixel readback can answer "did anything appear" —
	 * a computed `background-color` reads back the declared value whether or not it is visible.
	 */
	test('a ghost icon button on a content toolbar paints a visible hover fill', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId = await solePanelId(page);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, uid);
		try {
			// The link is stored as the node UID (stable across rename), not its display name.
			await page.evaluate(
				([id, u]) => {
					(window as any).goofi.commands.setPanelType(id, 'parameters');
					(window as any).goofi.commands.bindNodeToPanel(id, u);
				},
				[panelId, uid] as const
			);

			const panel = page.getByTestId('node-linked-panel');
			const btn = panel.getByRole('button', { name: 'Unlink node' });
			await expect(btn).toBeVisible();

			// Free to assert here — this is the app's only linked-node header, and it is already up. The
			// name in it is the identifier the CANVAS paints on the node, so it reads in mono wherever it
			// appears (D-T3), exactly as `ParamForm`'s rename title does; the panel chrome around it is
			// sans. It is the node PICKER's selected option now — the face travels down to the <select>
			// from the wrapper through app.css's `font: inherit`, so the <select> is what to measure.
			expect(
				await panel
					.getByTestId('panel-node')
					.locator('select')
					.evaluate((el) => getComputedStyle(el).fontFamily),
				'the linked node’s name renders in the app mono face'
			).toContain('JetBrains Mono');

			const box = (await btn.boundingBox())!;
			// A 4px corner square, clear of the centred glyph, so this reads the FILL and nothing else.
			const corner = { x: box.x + 1, y: box.y + 1, width: 4, height: 4 };

			await page.mouse.move(5, 5);
			await page.waitForTimeout(200); // past the --dur-fast fill transition
			const rest = await meanPixel(page, corner);
			await btn.hover();
			await page.waitForTimeout(200);
			const hovered = await meanPixel(page, corner);

			expect(
				hovered - rest,
				'the ghost hover fill lifts the toolbar it sits on, instead of repainting it'
			).toBeGreaterThan(5);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
			await restoreEditor(page, panelId);
		}
	});
});

test.describe('the layout converges between tabs', () => {
	/**
	 * Two clients on one patch — a desktop and the phone next to it.
	 *
	 * The panel arrangement is the FIFTH CRDT doc root. Every gesture is a layout command the manager
	 * applies, mirrors and broadcasts as a delta, so a split on one screen appears on the other through
	 * the very machinery a node add already uses. Before the cutover this did not happen at all: the
	 * arrangement was the client's, pushed back as an opaque blob, and a peer merged what it could.
	 *
	 * The other half is what does NOT travel. Where a client is looking — its front tab, its focused
	 * panel, how deep into a sub-patch each editor has gone — is viewpoint, never a doc root, so a
	 * phone three levels in stays there while the desktop rearranges around it, and neither client
	 * writes the arrangement in answer to the other.
	 *
	 * Two browser CONTEXTS, not two pages: the session id that scopes undo lives in `sessionStorage`,
	 * which a second page in the same context would share. Both contexts reach the ONE backend this
	 * worker owns — the two clients converging on one manager is the whole point — so every test here
	 * hands the workspace back.
	 */

	/** Comfortably past a layout command and its delta coming back. */
	const PAST_DEBOUNCE = 1200;

	const tabs = (page: Page) => page.getByTestId('workspace-tabs').locator('.ui-tab');
	const panels = (page: Page) => page.locator('.panel');

	/** Count this page's OWN writes to the arrangement. Must be attached before `goto` — `websocket`
	 * only fires for sockets opened after the listener. */
	function countLayoutWrites(page: Page): () => number {
		let n = 0;
		page.on('websocket', (ws) => {
			if (!ws.url().endsWith('/control')) return;
			ws.on('framesent', (f) => {
				if (typeof f.payload !== 'string') return;
				if (/"(page|session)_[a-z_]+"/.test(f.payload)) n += 1;
			});
		});
		return () => n;
	}

	/** Two nodes grouped into a sub-patch, on the page that will own the teardown. */
	async function makeSubPatch(page: Page): Promise<{ osc: string; buf: string; inst: string }> {
		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);
		const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
		await waitForNode(page, buf);
		const inst: string = await page.evaluate(
			([o, b]) => (window as any).goofi.commands.groupNodes([o, b], [120, 120]),
			[osc, buf] as const
		);
		await expect(page.getByTestId('subpatch-node')).toBeVisible();
		return { osc, buf, inst };
	}

	test('a layout change in one tab converges in another', async ({ browser }) => {
		const ctxA = await browser.newContext();
		const ctxB = await browser.newContext();
		const a = await ctxA.newPage();
		const b = await ctxB.newPage();
		const writesB = countLayoutWrites(b);
		try {
			await a.goto('/');
			await waitForApp(a);
			await b.goto('/');
			await waitForApp(b);
			await expect(panels(b)).toHaveCount(1);
			const before = writesB();

			// A SPLIT, through the real header menu — the arrangement's own structure, not a tab. This is
			// the change that used to reach nobody.
			await splitRight(a);
			await expect(panels(b), 'the split A made arrived through the doc').toHaveCount(2);

			// A closes it again, and the close converges too — a delta, not a wholesale blob.
			await closeSplit(a);
			await expect(panels(b), '…and so does the close').toHaveCount(1);

			// A tab, too — and B stays on its OWN tab, because which one is in front is viewpoint.
			await a.evaluate(() => (window as any).goofi.commands.addTab());
			await expect(tabs(b), 'the tab A added arrived').toHaveCount(2);
			await expect(
				tabs(b).first(),
				'…and B is still looking at its own tab, not the one A brought to the front'
			).toHaveAttribute('aria-selected', 'true');
			await expect(tabs(a).nth(1), 'while A is on the tab it just made').toHaveAttribute(
				'aria-selected',
				'true'
			);

			// No echo, no storm: converging on a peer's arrangement must write nothing back. Waited well
			// past the round trip, so a write that was merely slow would still be counted.
			await b.waitForTimeout(PAST_DEBOUNCE);
			expect(writesB() - before, 'B must not write the arrangement back at the manager').toBe(0);
		} finally {
			if ((await tabs(a).count()) > 1) await closeAddedTab(a);
			await ctxA.close();
			await ctxB.close();
		}
	});

	test('navigating does not move the other client, and a peer’s edit does not move us', async ({
		browser
	}) => {
		const ctxA = await browser.newContext();
		const ctxB = await browser.newContext();
		const a = await ctxA.newPage();
		const b = await ctxB.newPage();
		let made: { osc: string; buf: string; inst: string } | null = null;
		try {
			await a.goto('/');
			await waitForApp(a);
			await b.goto('/');
			await waitForApp(b);

			made = await makeSubPatch(a);
			await expect(b.getByTestId('subpatch-node'), 'the sub-patch reached B through the doc').toBeVisible();

			// B descends into it. That is navigation — where B is looking — so A must stay put.
			await b.getByTestId('subpatch-node').dblclick();
			await expect(b.getByTestId('subpatch-breadcrumb'), 'B is inside the sub-patch').toBeVisible();
			await b.waitForTimeout(PAST_DEBOUNCE);
			await expect(
				a.getByTestId('subpatch-breadcrumb'),
				'A must not be dragged into a sub-patch B entered'
			).toBeHidden();

			// Now A AUTHORS. The tab travels; B's own position inside the sub-patch survives it.
			await a.evaluate(() => (window as any).goofi.commands.addTab());
			await expect(tabs(b), 'A’s new tab arrived').toHaveCount(2);
			await expect(
				b.getByTestId('subpatch-breadcrumb'),
				'…and B is still where it was looking'
			).toBeVisible();
			await expect(tabs(b).first(), 'still on its own tab, too').toHaveAttribute(
				'aria-selected',
				'true'
			);
		} finally {
			// B climbs out first — its editor must not be sitting inside an instance about to dissolve —
			// then A hands the tab back LAST, so the arrangement the manager keeps is the pristine one.
			const crumb = b.getByTestId('subpatch-breadcrumb');
			if (await crumb.isVisible())
				await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
			await b.waitForTimeout(PAST_DEBOUNCE);
			if ((await tabs(a).count()) > 1) await closeAddedTab(a);
			if (made) {
				await a.evaluate((i) => (window as any).goofi.commands.expandInstance(i), made.inst);
				await a.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [
					made.osc,
					made.buf
				]);
				await waitForNoNode(a, made.osc);
				await waitForNoNode(a, made.buf);
			}
			await ctxA.close();
			await ctxB.close();
		}
	});
});

test.describe('which layout writes dirty the patch', () => {
	/**
	 * The dirty taxonomy (R spec §4 / D-R3): *navigation* must not mark the patch unsaved, *authoring*
	 * must. Since the arrangement became the manager's, the two halves are different OPS — authoring is
	 * a layout command, navigation is `set_viewpoint` — so the taxonomy holds by construction rather
	 * than by a flag the client sets. The assertions below are unchanged across that move, which is the
	 * point of having them: they are about the dot, not about the mechanism.
	 *
	 * The waits stay too. They were sized for a 400ms debounce that no longer exists, so they are now
	 * only slack — and slack is what makes "still clean" mean something rather than "not yet". Each
	 * test leaves the workspace as it found it, because a worker's specs share one backend.
	 */

	/** Comfortably past the write and its round trip. */
	const PAST_DEBOUNCE = 1200;

	let scratch = '';
	const patchName = `dirty-taxonomy-${process.pid}-${Date.now()}.gfi`;

	test.beforeAll(() => {
		scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-dirty-')));
	});
	test.afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }));

	/** Both tests below NAME the patch (`saveClean`), and the name is the manager's — it outlives the
	 * page and turns every later spec's Save into a silent overwrite of this file. Their own resets are
	 * the last statement of the body, which a red above never reaches, so one failure here cascades
	 * `a previous spec left the patch NAMED` through the rest of the run. Hand it back regardless,
	 * matching `touch-authoring.spec.ts`. */
	test.afterEach(async ({ page }) => {
		await page.evaluate(() => (window as any).goofi.commands.newPatch()).catch(() => {});
	});

	function unsavedChanges(page: Page): Promise<boolean> {
		return page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges);
	}

	/** Save the patch to the scratch file and wait for the manager to report it clean. */
	async function saveClean(page: Page): Promise<void> {
		await page.evaluate((p) => (window as any).goofi.commands.save(p), path.join(scratch, patchName));
		await expect.poll(() => unsavedChanges(page), { message: 'a save makes it clean' }).toBe(false);
	}

	function firstPanelId(page: Page): Promise<string> {
		return page.evaluate(() => (window as any).goofi.query.panels()[0].panelId);
	}

	test('entering and leaving a sub-patch never dirties the patch', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.

		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);
		const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
		await waitForNode(page, buf);
		const inst: string = await page.evaluate(
			([a, b]) => (window as any).goofi.commands.groupNodes([a, b], [120, 120]),
			[osc, buf] as const
		);
		const group = page.getByTestId('subpatch-node');
		await expect(group, 'the two nodes became one sub-patch facade').toBeVisible();

		await saveClean(page);

		// ENTER — the real door: a double-click on the group node.
		await group.dblclick();
		const crumbs = page.getByTestId('subpatch-breadcrumb');
		await expect(crumbs, 'the editor descended into the sub-patch').toBeVisible();
		await page.waitForTimeout(PAST_DEBOUNCE);
		expect(await unsavedChanges(page), 'entering a sub-patch is navigation, not an edit').toBe(false);

		// LEAVE — the breadcrumb back to the top level. Same axis, same answer.
		await crumbs.getByRole('button', { name: 'Patch', exact: true }).click();
		await expect(crumbs, 'the editor climbed back out').toBeHidden();
		await page.waitForTimeout(PAST_DEBOUNCE);
		expect(await unsavedChanges(page), 'leaving a sub-patch is navigation too').toBe(false);

		// Dissolve the facade before wiping: `removeNodes` takes the members with it but leaves the
		// empty instance behind, and a leaked sub-patch is a second `subpatch-node` for the next run.
		await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
		await expect(group, 'the sub-patch facade is gone').toHaveCount(0);
	});

	test('changing a docked viewer’s type DOES dirty the patch', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.

		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);

		// Borrow the single panel as a Viewer bound to the oscillator, then save — so the ONLY thing
		// left to change is the viewer type itself.
		const panelId = await firstPanelId(page);
		await page.evaluate(
			([pid, uid]) => {
				(window as any).goofi.commands.setPanelType(pid, 'viewer');
				(window as any).goofi.commands.bindNodeToPanel(pid, uid);
			},
			[panelId, osc] as const
		);
		const kind = page.getByTestId('viewer-kind').locator('select');
		await expect(kind, 'the panel is showing the oscillator with its type dropdown').toBeVisible();
		// Let the SETUP's own push land before saving. Both calls above are authoring, and the folded
		// intent only resets when the debounced push takes it — so without this wait a save clears the
		// flag while an `authored` is still pending, and the push that lands ~167ms later dirties the
		// patch no matter how the viewer-kind write is classified. The assertion below would then be
		// green against a viewer kind reclassified as navigation, which is the one thing it exists to
		// catch.
		await page.waitForTimeout(PAST_DEBOUNCE);
		await saveClean(page);

		await kind.selectOption('image');
		await expect
			.poll(() => unsavedChanges(page), { message: 'picking a viewer type is authoring' })
			.toBe(true);

		// Hand the workspace back: the type swap discards the panel's viewer state with it.
		await page.evaluate(
			(pid) => (window as any).goofi.commands.setPanelType(pid, 'node-editor'),
			panelId
		);
		await page.waitForTimeout(PAST_DEBOUNCE);
	});
});
