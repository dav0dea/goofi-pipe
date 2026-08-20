/**
 * Touch, in one place: the doors a coarse pointer needs, the floors every target has to meet, and
 * the gestures that only exist here.
 *
 * The rule the whole file is built on — orientation picks the ANCHOR, input MODALITY picks the
 * gesture and the affordance — is why this runs in the portrait `touch` project alone. What the
 * coarse media query answers is identical at 412px and at 1080px, so re-running it in three
 * projects would triple the wall clock to re-measure a constant. What genuinely differs by geometry
 * is `touch-reflow.spec.ts`; what must survive the orientation change is `touch-anchor.spec.ts`.
 */

import { test, expect, type Locator, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import {
	bareSpot,
	clearStage,
	inConn,
	inputLabel,
	outHandle,
	proximityStage
} from '../lib/cableDrag';
import { controlsInset } from '../lib/editor';
import { settledBox } from '../lib/geometry';
import {
	addErroringNode,
	addGlobal,
	addNode,
	nodeParams,
	nodes,
	tapNode,
	waitForNoNode,
	waitForNode
} from '../lib/goofi';
import { dismiss, spawnSh } from '../lib/harness';
import {
	dropNode as drop,
	editorHost as editor,
	openInspector as addAndSelect,
	pane,
	paneAxis
} from '../lib/inspector';
import { barsMatchTheHeader, controlsSitAtOneGap } from '../lib/panelBar';
import { openAddMenuByPress, paletteItem, openGhost } from '../lib/placement';
import { settledBar } from '../lib/topbar';
import {
	emptySpot,
	lowEmptySpot,
	touchSession,
	setKeyboardInset,
	type TouchPoint
} from '../lib/touch';
import { VIEWER_HOVER_SURFACES, hoverSettled, surfaceStyles, unhover } from '../lib/viewerChrome';

test.describe('the coarse tap-target floor, across the app\u2019s real chrome', () => {
	/**
	 * The PRODUCT hit-floor sweep (R spec §5.3, §6). `touch-ui-gallery.spec.ts` proves every `$lib/ui`
	 * primitive meets `--hit` at `/dev/ui`; this proves the app's real chrome does — where a product
	 * class rule routinely out-specifies the floor and silently defeats it.
	 *
	 * Two things it does that the older assertions did not (C6):
	 *
	 *  1. **Both axes.** A control 200px wide and 20px tall passed a height-only assertion. Several did.
	 *  2. **The real tap target, not the painted box.** A chrome strip shorter than `--hit` states its
	 *     compact box and lets an absolutely-positioned `::after` carry the target outward (the
	 *     IconButton pattern). `boundingBox()` cannot see that, so the effective rect below is the union
	 *     of the element's border box with any positioned, pointer-taking `::before`/`::after`.
	 *
	 * Measured, not hit-tested: size is the question here, reachability is elsewhere. (This once read
	 * "the TopBar's actions are pushed off the right edge, so a hit test would report Task 6's bug at
	 * every site" — Task 6 has since spilled them into the header's overflow menu instead, and
	 * `topbar-overflow.spec.ts` owns proving they are reachable there.)
	 *
	 * The registry is the extension point — R Task 9 adds rows (and the landscape/tablet projects) here
	 * rather than writing another bespoke spec.
	 */

	const HIT = 44;

	/** The effective tap rect: the border box, grown by any pseudo-element that takes pointer events —
	 *  unless the element clips its own overflow, in which case the pseudo cannot reach past the box.
	 *  That clip is not a detail: it is exactly what made the tab strip's ✕ untappable (C17). Ancestor
	 *  clipping is out of scope here; the "stays inside its host" test below is what covers that. */
	async function hitBox(control: Locator): Promise<{ width: number; height: number }> {
		return control.evaluate((el) => {
			const r = el.getBoundingClientRect();
			const own = getComputedStyle(el);
			// A pseudo's `inset`/`top`/`left` resolve against its containing block — the element's
			// PADDING box — and come back from getComputedStyle as used px.
			const padX = r.left + parseFloat(own.borderLeftWidth);
			const padY = r.top + parseFloat(own.borderTopWidth);
			let [left, top, right, bottom] = [r.left, r.top, r.right, r.bottom];
			for (const p of ['::before', '::after']) {
				const cs = getComputedStyle(el, p);
				if (cs.content === 'none' || cs.position === 'static' || cs.pointerEvents === 'none') continue;
				const [x, y, w, h] = [cs.left, cs.top, cs.width, cs.height].map(parseFloat);
				if (![x, y, w, h].every(Number.isFinite)) continue;
				if (own.overflowX === 'visible') {
					left = Math.min(left, padX + x);
					right = Math.max(right, padX + x + w);
				}
				if (own.overflowY === 'visible') {
					top = Math.min(top, padY + y);
					bottom = Math.max(bottom, padY + y + h);
				}
			}
			return { width: right - left, height: bottom - top };
		});
	}

	type Teardown = () => Promise<void>;

	interface Site {
		name: string;
		/** Reveal the control; returns the restore. The suite shares ONE backend and AppShell pushes the
		 *  layout into the running patch, so anything that touches the workspace must hand it back. */
		setup?: (page: Page) => Promise<Teardown>;
		control: (page: Page) => Locator;
	}

	/** A borrowed workspace is given back by a command, so the restore has landed as soon as the panel
	 *  redraws — this is the one beat that lets the redraw happen. */
	const settleLayout = (page: Page): Promise<void> => page.waitForTimeout(50);

	const firstPanelId = (page: Page): Promise<string> =>
		page.evaluate(() => (window as any).goofi.query.panels()[0].panelId);

	async function borrowPanel(page: Page, type: string): Promise<Teardown> {
		const panelId = await firstPanelId(page);
		await page.evaluate(
			([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
			[panelId, type] as const
		);
		return async () => {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			await settleLayout(page);
		};
	}

	const SITES: Site[] = [
		{
			name: 'the panel header content dropdown',
			control: (p) => p.getByTestId('panel-header').first().locator('.content-btn')
		},
		{
			name: 'the panel header maximize',
			control: (p) => p.getByTestId('panel-header').first().getByRole('button', { name: 'Maximize panel' })
		},
		{
			name: 'the panel header close',
			control: (p) => p.getByTestId('panel-header').first().getByRole('button', { name: 'Close panel' })
		},
		{
			name: 'the tab strip ＋',
			control: (p) => p.getByTestId('workspace-tabs').getByRole('button', { name: 'New tab' })
		},
		{
			// Switching layout tabs is the strip's primary action; the pill is what takes that tap.
			name: 'a workspace tab',
			control: (p) => p.getByTestId('workspace-tabs').locator('.ui-tab').first()
		},
		{
			// C17. The ✕ only exists once a second tab does, and it is collapsed to zero width on a fine
			// pointer — the coarse door is what makes it a target at all.
			name: 'the tab strip ✕',
			setup: async (page) => {
				await page.evaluate(() => (window as any).goofi.commands.addTab());
				await expect(page.getByTestId('workspace-tabs').locator('.ui-tab')).toHaveCount(2);
				return async () => {
					// Closing the tab we added (the last one) is the restore.
					await page
						.getByTestId('workspace-tabs')
						.getByRole('button', { name: 'Close tab' })
						.last()
						.click();
					await expect(page.getByTestId('workspace-tabs').locator('.ui-tab')).toHaveCount(1);
					await settleLayout(page);
				};
			},
			control: (p) => p.getByTestId('workspace-tabs').getByRole('button', { name: 'Close tab' }).last()
		},
		{
			name: 'the split seam',
			setup: async (page) => {
				const header = page.getByTestId('panel-header').first();
				await header.click({ button: 'right' });
				await page.locator('.context-menu .item', { hasText: 'Split Right' }).click();
				await expect(page.locator('.panel')).toHaveCount(2);
				return async () => {
					await page
						.getByTestId('panel-header')
						.nth(1)
						.getByRole('button', { name: 'Close panel' })
						.click();
					await expect(page.locator('.panel')).toHaveCount(1);
					await settleLayout(page);
				};
			},
			control: (p) => p.locator('.splitter').first()
		},
		{
			name: 'the file browser ✕',
			setup: openFileBrowser,
			control: (p) => p.getByTestId('fs-browser').getByRole('button', { name: 'Close' })
		},
		{
			name: 'the file browser up-one-level',
			setup: openFileBrowser,
			control: (p) => p.getByTestId('fs-browser').getByRole('button', { name: 'Up one level' })
		},
		{
			name: 'the globals add-row name field',
			setup: (page) => borrowPanel(page, 'globals'),
			control: (p) => p.locator('input.name').first()
		},
		{
			// The SAME control the frozen node header pins compact — here in a docked panel header,
			// which is real chrome and takes the floor. Both halves of the Select primitive's hook, tested.
			name: 'the docked viewer-kind select',
			setup: dockAViewer,
			control: (p) => p.getByTestId('viewer-kind').locator('select')
		},
		{
			// The node picker — the one control every node-binding panel wears, and the only door onto a
			// binding a phone has. It is measured UNBOUND, which is the state it exists for.
			name: 'the panel node picker',
			setup: (page) => borrowPanel(page, 'metadata'),
			control: (p) => p.getByTestId('panel-node').locator('select')
		},
		{
			// A Chip took `--hit` flat until the toolbar it sits in became exactly --panel-header-h tall;
			// it wears the strip's dense box now and has to win the floor back through `density="chrome"`,
			// the same restatement IconButton and Select make. `out` is two glyphs, so nothing about its
			// CONTENT would reach 44 in either axis.
			name: 'the console stream chip',
			setup: (page) => borrowPanel(page, 'console'),
			control: (p) => p.getByTestId('console-panel').locator('.ui-chip').first()
		},
		{
			// …and the other control the shortened bar re-boxed: the linked panel's unlink ✕, 20px on a
			// fine pointer now instead of 28.
			name: 'the panel unlink ✕',
			setup: dockAViewer,
			control: (p) => p.getByTestId('node-linked-panel').getByRole('button', { name: 'Unlink node' })
		},
		{
			// The cog beside it, and the sole door to every per-viewer setting. R-Task 2/3 scoped its
			// carve-out to SlotViewer's frozen 24px header, but `--node-u` is a `:root` token, so the
			// 24px bound applied in BOTH hosts — including this 44px panel header, where both its
			// siblings take the floor.
			name: 'the docked viewer-settings cog',
			setup: dockAViewer,
			control: (p) => p.getByTestId('node-linked-panel').getByTestId('viewer-settings-cog')
		},
		{
			// SvelteFlow's own stylesheet sizes these 26×26 and the app's coarse floor was `min-height`
			// ONLY, so each was 26×44 — the one §3.1c site in neither the fixed nor the declined column,
			// and never measured because a comment beside its insets asserted the opposite. It is also
			// the last Fit door touch has: D-R6 deleted the global one on the argument that every node
			// editor carries its own, `f` is a chord, and `zoomOnDoubleClick` is off.
			name: 'a flow viewport control',
			control: (p) => p.locator('.svelte-flow__controls button').first()
		},
		{
			// The agent chip: one glyph and one digit, so its CONTENT never reaches the floor — and it
			// is, by the code's own comment beside it, the only door onto detach and kill from outside
			// the panel. The Task 2 report reasoned it was "the primitives' own, already pinned"; the
			// primitives floor HEIGHT, and this measured 34.19 × 44.
			name: 'the TopBar agent chip',
			setup: withAgent,
			control: (p) => p.getByTestId('topbar-agents')
		},
		{
			// The destructive half of the question that chip asks. `Kill` is four characters wide.
			name: 'the agent kill button',
			setup: openAgentQuestion,
			control: (p) => p.getByTestId('agent-kill')
		},
		{
			name: 'the inspector resize handle',
			setup: async (page) => {
				const uid = await addNode(page, 'Oscillator', 'inputs');
				await waitForNode(page, uid);
				await tapNode(page, uid);
				await expect(page.getByTestId('auto-side-panel')).toHaveClass(/open/);
				return async () => {
					await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
					await waitForNoNode(page, uid);
				};
			},
			control: (p) => p.getByTestId('panel-resize-handle')
		}
	];

	/** Open the Load browser. Dispatched in-page rather than tapped, because at 412px `Load…` has
	 *  spilled into the header's overflow menu (R-Task 6) and is `display: none` in the bar — reaching
	 *  it through the menu is `topbar-overflow.spec.ts`'s and `touch-authoring.spec.ts`'s question, not
	 *  this file's. What is measured here is the dialog's chrome once it IS open. */
	async function openFileBrowser(page: Page): Promise<Teardown> {
		await page.getByTestId('topbar-load').evaluate((el: HTMLElement) => el.click());
		await expect(page.getByTestId('fs-browser')).toBeVisible();
		return async () => {
			await page.keyboard.press('Escape');
			await expect(page.getByTestId('fs-browser')).toBeHidden();
		};
	}

	/** Run an agent, so the chip that counts them exists at all. `_sh` is the hidden test adapter — a
	 *  plain `/bin/sh` — so nothing here needs a harness installed; handing it back dismisses it. */
	async function withAgent(page: Page): Promise<Teardown> {
		const id = await spawnSh(page);
		await expect(page.getByTestId('topbar-agents')).toBeVisible();
		return async () => {
			await dismiss(page, id);
			await expect(page.getByTestId('topbar-agents')).toBeHidden();
		};
	}

	/** …and ask the chip's question, which is the shell's dialog and needs no agent panel open. */
	async function openAgentQuestion(page: Page): Promise<Teardown> {
		const end = await withAgent(page);
		await page.getByTestId('topbar-agents').tap();
		await page.locator('.context-menu .item').first().tap();
		await expect(page.getByTestId('agent-close-dialog')).toBeVisible();
		return async () => {
			await page.keyboard.press('Escape');
			await expect(page.getByTestId('agent-close-dialog')).toBeHidden();
			await end();
		};
	}

	/** Borrow the sole panel as a Viewer bound to a fresh Oscillator, so its header renders the shared
	 *  ViewerControls — the same component the node canvas hosts under its frozen pin. */
	async function dockAViewer(page: Page): Promise<Teardown> {
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		const panelId = await firstPanelId(page);
		const restorePanel = await borrowPanel(page, 'viewer');
		await page.evaluate(
			([id, u]) => (window as any).goofi.commands.bindNodeToPanel(id, u),
			[panelId, uid] as const
		);
		await expect(page.getByTestId('node-linked-panel')).toBeVisible();
		return async () => {
			await restorePanel();
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		};
	}

	for (const site of SITES) {
		test(`${site.name} meets the coarse tap target in BOTH axes`, async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const teardown = await site.setup?.(page);
			try {
				const control = site.control(page);
				await control.waitFor({ state: 'attached' });
				const box = await hitBox(control);
				expect(box.width, `${site.name}: tap-target width`).toBeGreaterThanOrEqual(HIT);
				expect(box.height, `${site.name}: tap-target height`).toBeGreaterThanOrEqual(HIT);
			} finally {
				await teardown?.();
			}
		});
	}

	/**
	 * The other end of the same rule (R spec §5.4): iOS force-zooms the page when a control under 16px
	 * takes focus, which on a canvas app means the user is dumped at 2× with no way back to their own
	 * layout. `app.css` floors `input, select, textarea` at 16px under the coarse idiom — at (0,0,1),
	 * which ANY product class rule out-specifies. Several did; one test each, because each needs its own
	 * panel or its own gesture and one long test times out where several short ones do not.
	 */
	const fontSize = (loc: Locator): Promise<number> =>
		loc.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));

	test('the globals add-row name field clears the focus-zoom threshold', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const restore = await borrowPanel(page, 'globals');
		try {
			const field = page.locator('input.name').first();
			await expect(field).toBeVisible();
			expect(await fontSize(field)).toBeGreaterThanOrEqual(16);
		} finally {
			await restore();
		}
	});

	/* The chrome half of the Select primitive's hook. Its canvas half keeps --fs-small by design — the
	   frozen-exception test at the bottom of this file is what pins that. */
	test('the docked viewer-kind select clears the focus-zoom threshold', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const restore = await dockAViewer(page);
		try {
			expect(
				await fontSize(page.getByTestId('viewer-kind').locator('select'))
			).toBeGreaterThanOrEqual(16);
		} finally {
			await restore();
		}
	});

	/* The node picker rides the same chrome density, and adds a mono face to it — a `font-family` on the
	   wrapper, which the <select>'s own `font: inherit` picks up. That is exactly the shape that could
	   drag a font-SIZE along with it, so the threshold is measured here too rather than assumed from the
	   kind picker's row above. */
	test('the panel node picker clears the focus-zoom threshold', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const restore = await borrowPanel(page, 'metadata');
		try {
			expect(await fontSize(page.getByTestId('panel-node').locator('select'))).toBeGreaterThanOrEqual(
				16
			);
		} finally {
			await restore();
		}
	});

	test('the fx multi-line editor clears the focus-zoom threshold', async ({ page }) => {
		// --fs-micro is ~10px, which the coarse floor has to lift. Driven through the real inspector:
		// add a node, tap it, turn fx on, and grow the editor — the doors a phone user has.
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			await tapNode(page, uid);
			const field = page.getByTestId('auto-side-panel').getByTestId('param-field-amplitude');
			await field.getByTestId('param-fx-toggle').tap();
			await field.getByTestId('param-expr-expand').tap();
			const ta = field.getByTestId('param-expr-multiline');
			await expect(ta).toBeVisible();
			expect(await fontSize(ta), 'textarea font-size >= 16 defeats iOS focus-zoom').toBeGreaterThanOrEqual(16);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});

	test('the inspector’s header rename field clears the focus-zoom threshold', async ({ page }) => {
		// --fs-strong is ~14px on a phone. The same door: the identity header's name opens an input.
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			await tapNode(page, uid);
			const form = page.getByTestId('auto-side-panel');
			await form.getByTestId('node-name').tap();
			const input = form.getByTestId('node-name-input');
			await expect(input).toBeVisible();
			expect(await fontSize(input), 'rename input font-size >= 16 defeats iOS focus-zoom').toBeGreaterThanOrEqual(16);
			await page.keyboard.press('Escape');
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});

	test('the layout-tab rename field clears the focus-zoom threshold', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await page.evaluate(() => (window as any).goofi.commands.addTab());
		const tabs = page.getByTestId('workspace-tabs');
		await expect(tabs.locator('.ui-tab')).toHaveCount(2);
		try {
			// Dispatched, not tapped: the tab is an HTML5 drag source, so a real double-click there also
			// runs the strip's own drag/reorder handling. What is under test is the field's geometry.
			const tab = tabs.locator('.ui-tab').last();
			await tab.dispatchEvent('dblclick');
			const rename = tab.locator('input.ui-tab-rename');
			await expect(rename).toBeVisible();
			expect(await fontSize(rename)).toBeGreaterThanOrEqual(16);
			await page.keyboard.press('Escape');
		} finally {
			await tabs.getByRole('button', { name: 'Close tab' }).last().click();
			await expect(tabs.locator('.ui-tab')).toHaveCount(1);
			await settleLayout(page);
		}
	});

	/**
	 * The fourth force-zoom site, and the one that ALSO overflowed: a boundary pill's inline rename
	 * declared 11px type (out-specifying app.css's floor) but no `min-height` (so it did NOT
	 * out-specify the 44px one) — an input 17px taller than the 26px pill around it.
	 */
	test('a boundary pill’s rename fits the pill and clears the focus-zoom threshold', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);
		const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
		await waitForNode(page, buf);
		// A link ACROSS the cut is what makes `groupNodes` mint a boundary pill inside the sub-patch.
		await page.evaluate(
			([o, b]) =>
				(window as any).goofi.commands.addLink({
					node_out: o,
					slot_out: 'out',
					node_in: b,
					slot_in: 'data'
				}),
			[osc, buf] as const
		);
		const inst: string = await page.evaluate(
			(b) => (window as any).goofi.commands.groupNodes([b], [200, 200]),
			buf
		);
		try {
			await page.getByTestId('subpatch-node').dblclick();
			await expect(page.getByTestId('subpatch-breadcrumb')).toBeVisible();

			const pill = page.locator('.boundary').first();
			await expect(pill).toBeVisible();
			const pillBox = (await pill.boundingBox())!;
			// Dispatched, not tapped: the pill is a SvelteFlow node, and a real double-click there also
			// runs the pane's own selection/zoom handling. What is under test is the field's geometry.
			await pill.locator('.lbl').dispatchEvent('dblclick');
			const edit = pill.locator('.lbl-edit');
			await expect(edit).toBeVisible();

			expect(await fontSize(edit), 'the pill rename field').toBeGreaterThanOrEqual(16);
			const editBox = (await edit.boundingBox())!;
			expect(
				editBox.height,
				'the rename field does not stand taller than the pill it edits'
			).toBeLessThanOrEqual(pillBox.height + 8);
			await page.keyboard.press('Escape');
		} finally {
			await page.getByTestId('subpatch-breadcrumb').getByRole('button', { name: 'Patch', exact: true }).click();
			await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
			await page.evaluate(
				(ids) => (window as any).goofi.commands.removeNodes(ids),
				[osc, buf] as const
			);
			await waitForNoNode(page, osc);
			await settleLayout(page);
		}
	});

	/**
	 * C16, the other side of the floor. A console row is one 16px text line on a fine pointer, but on
	 * touch every control it hosts is floored to `--hit`, so the row is `--hit` tall — which is right
	 * (the row is the tap target) and is exactly what the virtual scroller's `estimateH` did NOT know.
	 * `layout.cum` sums that estimate for every row the ResizeObserver has not reached, so the thumb
	 * was short by ~28px per row. This pins the MODEL against the DOM, like `console-rows.spec.ts`
	 * does on the other pointer.
	 */
	test('a console row on touch is what the scroller’s height model says it is', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const restore = await borrowPanel(page, 'console');
		// A node erroring on its empty required input (see `addErroringNode`); the graph store mirrors
		// the error into the console. The cheapest real console content.
		const uid = await addErroringNode(page);
		try {
			const row = page.getByTestId('console-entry').first();
			await expect(row, 'the node error reached the console').toBeVisible();

			const m = await row.evaluate((el) => {
				const cs = getComputedStyle(el);
				const h = (sel: string) => el.querySelector(sel)!.getBoundingClientRect().height;
				return {
					row: el.getBoundingClientRect().height,
					txt: h('.txt'),
					node: h('.node'),
					copy: h('.console-copy-btn'),
					pad: parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom),
					border: parseFloat(cs.borderBottomWidth) + parseFloat(cs.borderTopWidth),
					hit: parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hit'))
				};
			});
			expect(m.hit, 'the coarse floor is in force').toBe(44);
			// The cause: both in-row controls take the floor, so the ROW does — this is the term the
			// estimate was missing.
			expect(m.node, 'the node chip is floored to --hit').toBeGreaterThanOrEqual(m.hit);
			expect(m.copy, 'the copy button is floored to --hit').toBeGreaterThanOrEqual(m.hit);
			expect(
				m.row,
				'the row is max(text, --hit) + padding + border — exactly what estimateRowHeight models'
			).toBeCloseTo(Math.max(m.txt, m.hit) + m.pad + m.border, 0);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
			await restore();
		}
	});

	/**
	 * The frozen exceptions, stated (R spec §5.3 "with the frozen exceptions stated and commented").
	 * The node canvas is a fixed-px coordinate system — a slot header is exactly one `--node-u` (24px)
	 * and `.surface` clips it — so a control inside it CANNOT take the 44px floor. What it must do is
	 * stay inside the strip it sits in: `.tri` declared `height`, not `min-height`, so app.css's coarse
	 * floor still applied and stood a 44px button in a 24px header, clipped by the node surface.
	 *
	 * Compared against the header rather than a literal, because the canvas is zoomable and a raw px
	 * expectation would move with the viewport transform.
	 */
	test('the frozen slot-header controls stay inside the header they sit in', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			const header = page.locator(`.slot-viewer[data-node="${uid}"] header`).first();
			await header.waitFor();
			const hb = (await header.boundingBox())!;
			for (const [name, sel] of [
				['the disclosure triangle', '.tri'],
				['the viewer-settings cog', '[data-testid=viewer-settings-cog]'],
				['the viewer-kind select', '[data-testid="viewer-kind"] select']
			] as const) {
				const box = (await header.locator(sel).boundingBox())!;
				expect(box.height, `${name} does not outgrow the frozen slot header`).toBeLessThanOrEqual(
					hb.height + 0.5
				);
			}
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});
});

test.describe('the density floor itself', () => {
	// Under a coarse pointer, interactive controls must meet a 44px touch target.
	// Under a fine pointer (the default project) they must NOT be inflated — verified by the
	// absence of this project from the default run (testMatch scopes it to the `touch` project).
	test('coarse pointer floors control height at 44px', async ({ page }) => {
		// Wide enough that the header's Save is still IN the bar. Since R-Task 6 the actions spill into
		// the overflow menu as the width runs out, and at the phone's own 412px every one of them has —
		// a spilled action is `display: none`, which has no box to measure. The question here is the
		// coarse floor, not how much bar a phone has (the collapse keys on width, not device class, so
		// this viewport is as coarse as the default one).
		await page.setViewportSize({ width: 1000, height: 800 });
		await page.goto('/');
		// By testid, not text: Save is an icon button now (the header's one icon family).
		const btn = page.getByTestId('topbar-save');
		await btn.waitFor();
		const h = await btn.evaluate((el) => el.getBoundingClientRect().height);
		expect(h).toBeGreaterThanOrEqual(44);
		const barH = await page
			.locator('.topbar')
			.evaluate((el) => el.getBoundingClientRect().height);
		expect(barH, 'the natural-height bar grows with its coarse-pointer controls').toBe(h);
	});

	// The workspace chrome strips render deliberately dense icon buttons — the tab bar's ＋ at 22px
	// and the panel header's maximize/close at 20px, both under the IconButton `--hit` floor
	// (workspace-chrome.spec pins those fine-pointer numbers). The dense box is a FINE-pointer
	// affordance only: under a coarse pointer the floor must be restored so each is a real tap
	// target. That restore is `density="chrome"` in the IconButton primitive; this is its guard —
	// nothing else in the suite covers it, so a strip that re-pins its own box (or a primitive that
	// drops the floor) would otherwise ship a 20px touch target silently.
	test('the chrome-dense workspace icon buttons meet the 44px coarse tap target', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const add = page.getByTestId('workspace-tabs').getByRole('button', { name: 'New tab' });
		await add.waitFor();
		const abox = (await add.boundingBox())!;
		expect(abox.width, 'the tab-strip ＋ is a real tap target on touch').toBeGreaterThanOrEqual(44);
		expect(abox.height, 'the tab-strip ＋ is a real tap target on touch').toBeGreaterThanOrEqual(44);

		const max = page
			.getByTestId('panel-header')
			.first()
			.getByRole('button', { name: 'Maximize panel' });
		await max.waitFor();
		const mbox = (await max.boundingBox())!;
		expect(mbox.width, 'the header maximize is a real tap target on touch').toBeGreaterThanOrEqual(44);
		expect(mbox.height, 'the header maximize is a real tap target on touch').toBeGreaterThanOrEqual(
			44
		);
	});

	// The other end of the same rule. A slot header is FROZEN node-canvas geometry (`--node-u`, 24px,
	// mirrored in nodeMetrics.ts), so it is the one strip that cannot grow to hold a floored box: the
	// viewer-settings cog pins its VISUAL box under the floor and lets IconButton's coarse `::after`
	// carry the tap target instead. Routing it through `density="chrome"` would look like the same
	// seam the two buttons above use, and would silently stand a 44px cog inside a 24px header.
	// Compared against the header rather than a literal, so canvas zoom cannot move the goalposts.
	test('the viewer-settings cog stays inside the frozen slot header on touch', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);

		const header = page.locator(`.slot-viewer[data-node="${uid}"] header`).first();
		const cog = header.getByTestId('viewer-settings-cog');
		await cog.waitFor();
		const hbox = (await header.boundingBox())!;
		const cbox = (await cog.boundingBox())!;
		expect(cbox.height, 'the cog does not outgrow the frozen header it sits in').toBeLessThanOrEqual(
			hbox.height
		);

		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	});
});

test.describe('a door for every affordance hover used to own', () => {
	/**
	 * R Task 4 — no interaction and no information may exist SOLELY behind hover (CLAUDE.md).
	 *
	 * Two shapes of door are proved here, both under real touch:
	 *  - the app-wide `TitleTip` long-press layer, which is the ONE door built for the ~40 `title=`
	 *    tooltips (rather than 40 bespoke coarse cues), and
	 *  - the resting coarse form given to each control/label a fine pointer reveals on hover.
	 *
	 * Driven through CDP touch events like `touch-editor.spec.ts`: under `hasTouch` Playwright's mouse
	 * API still dispatches MOUSE events, and `pointerType: 'mouse'` is exactly the input the long-press
	 * layer stands down for — so a mouse-driven "press" would prove nothing.
	 */

	interface Pt {
		x: number;
		y: number;
	}

	async function touchSession(page: Page) {
		const cdp = await page.context().newCDPSession(page);
		const send = (type: string, touchPoints: Pt[]) =>
			cdp.send('Input.dispatchTouchEvent', { type, touchPoints } as never);
		return {
			down: (p: Pt) => send('touchStart', [p]),
			up: () => send('touchEnd', [])
		};
	}

	/** The centre of the first match, as a touch point. */
	async function centreOf(page: Page, selector: string): Promise<Pt> {
		const box = (await page.locator(selector).first().boundingBox())!;
		expect(box, `${selector} is on screen`).toBeTruthy();
		return { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height / 2) };
	}

	/** The panel header's maximize button: on screen at this geometry, and it ACTS on tap — while
	 * `maximizedPanelId` is provably not part of `WorkspaceState`, so nothing here persists.
	 *
	 * Named by its own test id, not as "the first `.hdr-btn`": the header's right end is a progressive
	 * overflow now, so which control is first there is a function of the panel's width. Reaching for
	 * an ordinal picked up Split Right instead, which both asked the wrong tooltip and — one test
	 * later — really split the panel and left the arrangement behind. */
	const MAX_BTN = '[data-testid="panel-maximize"]';

	test('a long press reveals a title, and does not fire the control it asked about', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const touch = await touchSession(page);
		const tip = page.getByTestId('title-tip');

		const at = await centreOf(page, MAX_BTN);
		await touch.down(at);
		await expect(tip, 'the press surfaces the title the finger is on').toHaveText('Maximize');

		await touch.up();
		await expect(tip, 'and releasing leaves the answer standing').toHaveText('Maximize');
		// `toggleMaximize` would have flipped this to "Restore" had the click gone through. Asking what
		// a control does must never be the same as using it.
		await expect(
			page.locator(MAX_BTN).first(),
			'the press asked a question; it did not press the button'
		).toHaveAttribute('title', 'Maximize');

		// The next press anywhere dismisses — including one that resolves no title at all.
		await touch.down({ x: at.x, y: at.y + 200 });
		await touch.up();
		await expect(tip).toHaveCount(0);
	});

	test('a tap is not a press — the control acts and no tooltip appears', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const touch = await touchSession(page);
		const btn = page.locator(MAX_BTN).first();

		const at = await centreOf(page, MAX_BTN);
		await touch.down(at);
		await touch.up();
		await expect(btn, "a tap is still the control's own action").toHaveAttribute('title', 'Restore');
		await expect(page.getByTestId('title-tip')).toHaveCount(0);

		// Put it back — one backend serves every spec on this worker.
		await touch.down(await centreOf(page, MAX_BTN));
		await touch.up();
		await expect(btn).toHaveAttribute('title', 'Maximize');
	});

	test('a node’s update rate is legible without hover', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// Buffer for the input slot (Oscillator has none); Oscillator for the rate (it is the one that
		// ticks unprompted, so it is the one that reports a rate).
		const buf = await addNode(page, 'Buffer', 'signal', [60, 60]);
		const osc = await addNode(page, 'Oscillator', 'inputs', [60, 260]);
		await waitForNode(page, buf);
		await waitForNode(page, osc);
		try {
			// The input NAME's coarse door used to be pinned here as "it rests open". It does not any
			// more: resting it open put a name tag on every input on the canvas at all times, so the door
			// is a proximity reveal scoped to a cable in flight — `touch-slot-name.spec.ts` owns both
			// halves of it, and this file keeps the rate, whose door really is a resting form.

			// The header rate sits at opacity .3 at rest and comes forward on hover — which on touch is
			// never. It appears with the node's first stats push.
			const rate = page.locator('.goofi-node .rate').first();
			await expect(rate).toBeVisible({ timeout: 15_000 });
			const op = await rate.evaluate((el) => parseFloat(getComputedStyle(el).opacity));
			expect(op, 'the rate is readable without a hover to bring it forward').toBeGreaterThan(0.8);
		} finally {
			for (const u of [buf, osc]) {
				await page.evaluate((n) => (window as any).goofi.commands.removeNode(n), u);
				await waitForNoNode(page, u).catch(() => {});
			}
		}
	});

	test('a tap on a plot yields the per-sample readout hover used to own', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			const touch = await touchSession(page);
			const plot = page.locator(`.slot-viewer[data-node="${uid}"] .container`).first();
			await expect(plot).toBeVisible();
			const chip = page.getByTestId('cursor-chip');
			await expect(chip, 'nothing is pinned before the tap').toHaveCount(0);

			const box = (await plot.boundingBox())!;
			const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height / 2) };
			await touch.down(at);
			// Asserted WHILE THE FINGER IS DOWN, which is the whole discriminator: the browser's
			// compatibility `mousemove` only arrives after the release, so a `mousemove`-driven readout
			// cannot exist during the gesture — exactly when a scrub needs it.
			await expect(chip, 'the press pins a sample while the finger is still on it').toBeVisible();

			// …and it SURVIVES the release. A touch pointer is destroyed on lift, firing `pointerleave`
			// immediately; retracting there would make the only per-sample readout a flash.
			await touch.up();
			await expect(chip, 'lifting the finger keeps the readout').toBeVisible();
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});

	test('the console per-row copy button is reachable without hover', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

		// Same content source as `console-rows.spec.ts`: a node erroring on its empty required input
		// (see `addErroringNode`), which the graph store mirrors into the console.
		const uid = await addErroringNode(page);
		try {
			const copy = page.getByTestId('console-copy').first();
			await expect(copy).toBeVisible();
			await expect(copy, 'the copy control rests visible where there is no hover').toHaveCSS(
				'opacity',
				'1'
			);
			const events = await copy.evaluate((el) => getComputedStyle(el).pointerEvents);
			expect(events, 'and it is tappable, not merely painted').not.toBe('none');

			// …and it is a GHOST, like every other chrome-density icon button in the app. Resting it
			// open was the right call; inheriting `.ui-icon-btn`'s filled + outlined base paint with it
			// was not. On a fine pointer that is invisible (16px at `opacity: 0`); under coarse it puts
			// a painted 44×44 box on every console row — the highest repetition rate in the app, against
			// app.css's own rule that a surface step is what carries separation.
			const paint = await copy.evaluate((el) => {
				const cs = getComputedStyle(el);
				return { border: cs.borderTopColor, bg: cs.backgroundColor };
			});
			const invisible = /rgba\(0, 0, 0, 0\)|transparent/;
			expect(paint.border, 'the resting copy button draws no border').toMatch(invisible);
			expect(paint.bg, 'nor a filled surface').toMatch(invisible);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			// The panel type is a command, and the assertion above only passes once the manager's own
			// delta drew it back — so the running patch already holds the restore.
		}
	});

	test('the inspector resize seam is visible without hover', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			await tapNode(page, uid);
			const handle = page.getByTestId('panel-resize-handle');
			await expect(handle).toBeVisible();
			const painted = await handle.evaluate(
				(el) => getComputedStyle(el, '::after').backgroundColor
			);
			expect(painted, 'the seam paints a line at rest, not only on hover').not.toBe(
				'rgba(0, 0, 0, 0)'
			);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});
});

test.describe('the canvas under a finger', () => {
	/**
	 * The node editor's coarse-pointer door for adding a node (R spec §3.2b).
	 *
	 * Adding a node had four routes: a port click, a canvas double-click, Tab, and the app header's
	 * ＋ Add node. None of the first three is reachable on a phone — no double-click, no keyboard, and
	 * a port click needs a node to already exist — and the header button is panel-local behaviour that
	 * this same change removed (it was clipped off-screen at 412px anyway). A long press on empty
	 * canvas is now the touch door, and it is the ONLY one, so it earns a gesture-level guard on top
	 * of the recognizer's unit tests.
	 *
	 * Driven through CDP touch events, not `page.mouse`: under `hasTouch` Playwright's mouse API still
	 * dispatches MOUSE events, whose `pointerType` is `mouse` — exactly the input this door is closed
	 * to, so a mouse-driven "long press" would prove nothing.
	 */

	test('a long press on empty canvas opens the add-node menu there', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const touch = await touchSession(page);
		const at = await emptySpot(page);
		const menu = page.getByTestId('add-node-menu-anchor');

		await touch.down(at);
		await expect(menu, 'the press opens the menu while the finger is still down').toBeVisible();

		// The touchend that ENDS the opening gesture must not read as a dismissal: it lands on the
		// menu's own click-catcher, which the press itself mounted mid-gesture.
		await touch.up();
		await expect(menu, 'and releasing that same press does not close it again').toBeVisible();

		const box = (await menu.boundingBox())!;
		const vp = page.viewportSize()!;
		// Anchored to the press vertically; horizontally the shared viewport clamp owns the answer at
		// 412px (a 320px menu cannot open at mid-screen and still fit), which is its whole job.
		expect(Math.abs(box.y - at.y), 'the menu opens at the finger, not at some centred fallback')
			.toBeLessThan(40);
		expect(box.x, 'and the clamp keeps it fully on screen').toBeGreaterThanOrEqual(0);
		expect(box.x + box.width).toBeLessThanOrEqual(vp.width);

		// Escape only reaches the menu through the editor's own keydown handler, which stands down
		// unless its panel is active — so this closing also proves the press left `Panel`'s
		// capture-phase `setActive` alone rather than swallowing the pointerdown that drives it.
		await page.keyboard.press('Escape');
		await expect(menu).toHaveCount(0);
	});

	/**
	 * The same door, in the geometry the test above deliberately avoids: a press LOW on the canvas.
	 *
	 * The menu opens under the finger, so when the press is within a menu's height of the bottom the
	 * shared clamp slides it UP — and the slid menu then covers the very point the finger is about to
	 * lift from. The one-shot swallow that keeps the opening release from dismissing the menu covered
	 * only the full-screen catcher underneath it, so in this geometry the release's compat click landed
	 * on the menu instead, hit a palette ROW, and left a placement ghost for a node type the user never
	 * chose. A press must open the menu and choose nothing.
	 *
	 * The press point is calibrated against the REAL menu height (the palette's contents decide it), not
	 * hardcoded — the assertions below would otherwise pass on a menu the clamp never touched.
	 */
	test('a long press low on the canvas does not pick the row under the finger', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const touch = await touchSession(page);
		const menu = page.getByTestId('add-node-menu-anchor');

		await openAddMenuByPress(page);
		const menuHeight = (await menu.boundingBox())!.height;
		await page.keyboard.press('Escape');
		await expect(menu).toHaveCount(0);

		const vh = page.viewportSize()!.height;
		const at = await lowEmptySpot(page, vh - menuHeight);
		await touch.down(at);
		await expect(menu, 'the press opened the menu').toBeVisible();

		const box = (await menu.boundingBox())!;
		expect(box.y, 'the clamp slid the menu up past the press point').toBeLessThan(at.y);
		expect(box.y + box.height, '…so the menu now covers it — the geometry under test').toBeGreaterThan(
			at.y
		);

		await touch.up();
		await expect(page.getByTestId('placement-ghost'), 'the release chose nothing').toHaveCount(0);
		await expect(menu, 'and the menu still stands, waiting for a real choice').toBeVisible();

		await page.keyboard.press('Escape');
		await expect(menu).toHaveCount(0);
	});

	/**
	 * The other half of `editor-controls.spec.ts`'s inset guard. The coarse `--hit` floor makes each
	 * control button 44px tall, so on a 412px phone the cluster is a slab — and it was drawn 35px in
	 * from both edges (a declared 20px plus Flow's own 15px panel margin), which put it well inside the
	 * canvas rather than in its corner. Under coarse it tucks to `--space-6`.
	 *
	 * Clearing the panel's corner grip used to be a term here too — it no longer is, because the grip
	 * is not rendered under this idiom at all (`Panel.svelte`, `touch-panel-split.spec.ts`). What is
	 * left is the tuck itself, which is the whole point on a phone: the cluster belongs in the corner,
	 * not floating 35px inside a 412px canvas.
	 */
	test('the editor controls tuck into the corner under a coarse pointer', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const { left, bottom, rem } = await controlsInset(page);
		expect(left).toBeCloseTo(0.75 * rem, 0);
		expect(bottom).toBeCloseTo(0.75 * rem, 0);
		expect(left, 'a real tuck, not the fine-pointer inset').toBeLessThan(1.5 * rem);
	});

	test('a pan is not a press — dragging the canvas opens nothing', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const touch = await touchSession(page);
		const at = await emptySpot(page);

		await touch.down(at);
		await touch.moveTo({ x: at.x + 60, y: at.y + 20 });
		// Hold well past the recognizer's window with the finger parked at its new spot.
		await page.waitForTimeout(900);
		await expect(page.getByTestId('add-node-menu-anchor'), 'a drag never arms the door').toHaveCount(
			0
		);
		await touch.up();
	});

	/**
	 * Entering a sub-patch, the coarse half (the desktop half is `subpatch-entry.spec.ts`).
	 *
	 * `enterInstance`'s one caller is a hand-rolled double-click recogniser whose slop was `6` CSS px —
	 * a 12×12 box, an order of magnitude under any platform's tap slop, so a finger simply could not
	 * hit it twice. R added the CREATE half of sub-patches to touch (`Group into sub-patch` in the
	 * header's overflow menu), so touch could mint a sub-patch it had no way back into. The slop is
	 * chosen per GESTURE off `pointerType`, the same seam the long-press door reads — not per device.
	 */

	/** The sub-patch instances the patch currently holds. */
	const instances = (page: Page): Promise<Record<string, unknown>> =>
		page.evaluate(() => (window as any).goofi.query.instances());

	/** A one-member sub-patch at `at`, plus its member uid so the test can put both back. */
	async function groupOneNodeAt(
		page: Page,
		at: [number, number]
	): Promise<{ member: string; inst: string }> {
		const member = await addNode(page, 'Buffer', 'signal', at);
		await waitForNode(page, member);
		const inst: string = await page.evaluate(
			([m, p]) => (window as any).goofi.commands.groupNodes([m], p),
			[member, at] as const
		);
		await expect(page.locator(`.svelte-flow__node[data-id="${inst}"]`)).toBeVisible();
		return { member, inst };
	}

	async function dissolveAndRemove(page: Page, member: string, inst: string): Promise<void> {
		if (Object.keys(await instances(page)).includes(inst))
			await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
		await page.evaluate((m) => (window as any).goofi.commands.removeNode(m), member);
		await waitForNoNode(page, member).catch(() => {});
	}

	const centreOf = async (page: Page, uid: string): Promise<{ x: number; y: number }> => {
		const b = (await page.locator(`.svelte-flow__node[data-id="${uid}"]`).boundingBox())!;
		return { x: Math.round(b.x + b.width / 2), y: Math.round(b.y + b.height / 2) };
	};

	test('a double TAP enters a sub-patch, drift and all, and leaves it standing', async ({ page }) => {
		// The inspector's 120ms slide is what has to be OVER the node when the second tap lands, and a
		// race against a transition is not what this test is about. The app's own reduced-motion rule
		// collapses it to 0.01ms, so the pane's position is a function of the selection alone.
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const { member, inst } = await groupOneNodeAt(page, [80, 80]);
		try {
			const touch = await touchSession(page);
			const at = await centreOf(page, inst);
			await touch.down(at);
			await touch.up();
			await page.waitForTimeout(120);
			// 8px of drift — nothing on a finger, and twice the old 6px window.
			await touch.down({ x: at.x + 8, y: at.y + 4 });
			await touch.up();

			await expect(
				page.getByTestId('subpatch-breadcrumb'),
				'the second tap entered the sub-patch'
			).toBeVisible();
			// The first tap raises the inspector across all but --hit of a 412px editor, so the second
			// lands on the pane — including, for a sub-patch, its Expand (dissolve) button.
			await page.waitForTimeout(400);
			expect(
				Object.keys(await instances(page)),
				'and the pane it landed on did not also act on it'
			).toContain(inst);
		} finally {
			const crumb = page.getByTestId('subpatch-breadcrumb');
			if (await crumb.isVisible())
				await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
			await dissolveAndRemove(page, member, inst);
		}
	});

	/**
	 * The other half of that identity check, and the half the guard did not cover.
	 *
	 * `instanceUnder` answers `''` for two situations it cannot tell apart — no node under the pointer
	 * at all (the inspector-slid-over-the-node case the guard exists for) and *a node that is simply
	 * not a sub-patch instance*, since `g.instances` holds only instances. Both collapsed into the
	 * accept branch, so under the widened touch slop a second tap landing on an adjacent ORDINARY node
	 * was taken as the gesture's second half: the editor navigated into the sub-patch, and because a
	 * recognised second click is consumed in the capture phase, the node the user actually aimed at
	 * was never even selected.
	 */
	test('a tap on a plain neighbour is that node’s tap, not the sub-patch’s second', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		// Same reason as the sibling case below: with the inspector off nothing slides over the second
		// node, so this measures the identity guard and only the identity guard.
		await page.getByTestId('inspector-toggle').click();
		await expect(page.getByTestId('auto-side-panel')).not.toHaveClass(/open/);
		await expect(page.getByTestId('auto-side-panel')).toHaveCSS('visibility', 'hidden');
		const a = await groupOneNodeAt(page, [40, 60]);
		const plain = await addNode(page, 'Oscillator', 'inputs', [40, 60]);
		await waitForNode(page, plain);
		try {
			const ab = (await page.locator(`.svelte-flow__node[data-id="${a.inst}"]`).boundingBox())!;
			const scale = await page
				.locator('.svelte-flow__viewport')
				.first()
				.evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).a);
			// Parked BELOW rather than beside. Beside it, 14px past the seam lands in the neighbour's
			// left-edge slot handles, and a tap measured there did not select the node at all — which
			// would have greened the second assertion below by never delivering the tap. On the vertical
			// seam both points are node body.
			await page.evaluate(
				([u, p]) => (window as any).goofi.commands.setNodePos(u, p),
				[plain, [40, Math.round(60 + (ab.height + 2) / scale)] as [number, number]] as const
			);
			const p1 = { x: Math.round(ab.x + ab.width / 2), y: Math.round(ab.y + ab.height - 6) };
			const p2 = { x: p1.x, y: p1.y + 14 };
			await expect
				.poll(() =>
					page.evaluate(
						(pts) =>
							pts.map((p) =>
								(document.elementFromPoint(p.x, p.y) as HTMLElement | null)
									?.closest('.svelte-flow__node')
									?.getAttribute('data-id')
							),
						[p1, p2]
					)
				)
				.toEqual([a.inst, plain]);

			const touch = await touchSession(page);
			await touch.down(p1);
			await touch.up();
			await page.waitForTimeout(120);
			await touch.down(p2);
			await touch.up();
			await page.waitForTimeout(300);
			await expect(
				page.getByTestId('subpatch-breadcrumb'),
				'a plain node is not the sub-patch’s second tap'
			).toHaveCount(0);
			expect(
				await page.evaluate(() => (window as any).goofi.query.selection().nodes),
				'…and its tap was not swallowed either — it selected the node it landed on'
			).toEqual([plain]);
		} finally {
			const crumb = page.getByTestId('subpatch-breadcrumb');
			if (await crumb.isVisible())
				await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), plain);
			await waitForNoNode(page, plain).catch(() => {});
			await dissolveAndRemove(page, a.member, a.inst);
		}
	});

	test('two taps on two DIFFERENT sub-patches are not one gesture', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// With the inspector off, nothing slides over the second node — so this measures the slop and
		// only the slop, which is what widening it put at risk.
		await page.getByTestId('inspector-toggle').click();
		await expect(page.getByTestId('auto-side-panel')).not.toHaveClass(/open/);
		await expect(page.getByTestId('auto-side-panel')).toHaveCSS('visibility', 'hidden');
		const a = await groupOneNodeAt(page, [40, 60]);
		const b = await groupOneNodeAt(page, [40, 260]);
		try {
			// Park B's left edge just past A's right edge, so two points 14px apart — inside the coarse
			// slop — sit on two different nodes.
			const ab = (await page.locator(`.svelte-flow__node[data-id="${a.inst}"]`).boundingBox())!;
			const scale = await page
				.locator('.svelte-flow__viewport')
				.first()
				.evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).a);
			await page.evaluate(
				([u, p]) => (window as any).goofi.commands.setNodePos(u, p),
				[b.inst, [Math.round(40 + (ab.width + 2) / scale), 60] as [number, number]] as const
			);
			const p1 = { x: Math.round(ab.x + ab.width - 6), y: Math.round(ab.y + ab.height / 2) };
			const p2 = { x: p1.x + 14, y: p1.y };
			await expect
				.poll(() =>
					page.evaluate(
						(pts) =>
							pts.map((p) =>
								(document.elementFromPoint(p.x, p.y) as HTMLElement | null)
									?.closest('.svelte-flow__node')
									?.getAttribute('data-id')
							),
						[p1, p2]
					)
				)
				.toEqual([a.inst, b.inst]);

			const touch = await touchSession(page);
			await touch.down(p1);
			await touch.up();
			await page.waitForTimeout(120);
			await touch.down(p2);
			await touch.up();
			await page.waitForTimeout(300);
			await expect(
				page.getByTestId('subpatch-breadcrumb'),
				'a tap on a NEIGHBOUR is that neighbour’s first tap, not this one’s second'
			).toHaveCount(0);
			expect(
				await page.evaluate(() => (window as any).goofi.query.selection().nodes),
				'…and it really WAS delivered as a first tap — otherwise this asserts a non-event'
			).toEqual([b.inst]);
		} finally {
			const crumb = page.getByTestId('subpatch-breadcrumb');
			if (await crumb.isVisible())
				await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
			await dissolveAndRemove(page, a.member, a.inst);
			await dissolveAndRemove(page, b.member, b.inst);
		}
	});
});

test.describe('double-tap zoom', () => {
	/**
	 * DOUBLE-TAP-AND-DRAG TO ZOOM — the one-handed gesture every map app has, ADDED beside pinch.
	 *
	 * Two fingers were the only way to zoom this canvas, which on a phone means putting down whatever
	 * you are holding. The map gesture is the standard answer: tap, then tap again and keep the finger
	 * down, and dragging it moves the zoom — with the point you tapped held still under it, so you are
	 * magnifying the thing you pointed at rather than the middle of the screen.
	 *
	 * WHAT THIS FILE HAS TO PIN, and why each half regresses on its own:
	 *
	 *  · THE ZOOM ITSELF. `.svelte-flow__viewport`'s matrix is the one place the answer lives, and its
	 *    SCALE is the only part of it a pan cannot move. Read before and after.
	 *
	 *  · THE ANCHOR. A zoom about the screen centre also changes the scale, so the scale assertion
	 *    alone cannot tell an anchored zoom from an unanchored one. What can: the flow point drawn
	 *    under the tapped screen point, computed from the matrix before and after — it is the same
	 *    point if and only if the zoom was taken about it.
	 *
	 *  · THAT ONE FINGER STILL PANS. A recognizer that swallows the FIRST touch takes panning with it,
	 *    and nothing else in this suite would notice: a canvas that no longer pans still looks like a
	 *    canvas. So the guard is a plain tap-drag, asserting the matrix TRANSLATED and its scale did
	 *    not move — the mirror image of the assertions above.
	 *
	 * Pinch is untouched (`zoomOnPinch` is still SvelteFlow's own), and this gesture is deliberately
	 * additive: the seam it uses is `zoomOnDoubleClick={false}`, i.e. a double tap that was already
	 * inert.
	 */

	test.beforeEach(async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await clearGraph(page);
	});

	/** Hand the shared backend back empty even on failure — a leftover card is a tap target the next
	 *  spec's `emptySpot` has to work around. */
	test.afterEach(async ({ page }) => {
		await clearGraph(page).catch(() => {});
	});

	async function clearGraph(page: Page): Promise<void> {
		await page.evaluate(() => {
			const g = (window as any).goofi;
			const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
			if (uids.length) return g.commands.removeNodes(uids);
		});
		await expect.poll(async () => (await nodes(page)).length).toBe(0);
	}

	/** The pan/zoom matrix every flow-space thing is drawn through. */
	const viewport = (page: Page): Locator => page.locator('.svelte-flow__viewport');
	/** Its untransformed parent — so the pane's own top-left IS the matrix's origin. */
	const pane = (page: Page): Locator => page.locator('.svelte-flow__pane').first();

	interface Matrix {
		/** The zoom. */
		k: number;
		x: number;
		y: number;
	}

	async function matrixOf(page: Page): Promise<Matrix> {
		return viewport(page).evaluate((el) => {
			const m = new DOMMatrix(getComputedStyle(el).transform);
			return { k: m.a, x: m.e, y: m.f };
		});
	}

	const scaleOf = async (page: Page): Promise<number> => (await matrixOf(page)).k;

	/** The FLOW point drawn under a screen point — the matrix, inverted. */
	function flowUnder(m: Matrix, origin: TouchPoint, at: TouchPoint): TouchPoint {
		return { x: (at.x - origin.x - m.x) / m.k, y: (at.y - origin.y - m.y) / m.k };
	}

	/** Feed a finger from `at` to `at + (0, dy)` in steps, then let it come to rest. */
	async function dragBy(
		page: Page,
		touch: { moveTo(p: TouchPoint): Promise<unknown> },
		at: TouchPoint,
		dy: number
	): Promise<void> {
		for (let i = 1; i <= 5; i++) {
			await touch.moveTo({ x: at.x, y: at.y + Math.round((dy * i) / 5) });
		}
		// Chromium reads back-to-back synthetic moves as a FLING, which then eats the next tap anywhere
		// on the page. Coming to rest before the lift is what keeps this spec from poisoning the next.
		await page.waitForTimeout(150);
	}

	test('a double tap and a drag zooms the canvas, about the point that was tapped', async ({
		page
	}) => {
		const box = (await pane(page).boundingBox())!;
		const origin = { x: box.x, y: box.y };
		// Low in the pane, so the upward drag has room without leaving the screen. Asserted bare rather
		// than assumed: a tap that lands on a node card is a node drag, and would green nothing.
		const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height * 0.75) };
		await expect
			.poll(() =>
				page.evaluate(
					(p) => document.elementFromPoint(p.x, p.y)?.classList.contains('svelte-flow__pane'),
					at
				))
			.toBe(true);

		const before = await matrixOf(page);
		const held = flowUnder(before, origin, at);

		const touch = await touchSession(page);
		await touch.down(at);
		await touch.up();
		// The second tap, then HOLD, then drag — the gesture as a hand actually performs it, and the
		// hold is not incidental: 600 ms is past the 500 ms long press, so this is the arrangement in
		// which the add-node menu would open under the finger if the gesture did not disarm it.
		await touch.down(at);
		await page.waitForTimeout(600);
		await dragBy(page, touch, at, -Math.min(200, Math.round(box.height * 0.4)));
		await touch.up();

		await expect
			.poll(() => scaleOf(page), { message: 'dragging up zoomed in' })
			.toBeGreaterThan(before.k * 1.5);

		// The two things a finger held on this canvas ALSO means, neither of which may fire here: the
		// long press that is the coarse door onto the add-node menu (armed on the very `pointerdown`
		// that starts this gesture), and the compat `dblclick` a double tap would otherwise replay onto
		// the pane, which opens the same menu by the mouse route.
		await expect(page.getByTestId('add-node-menu-anchor'), 'zooming is not asking for a node').toHaveCount(0);

		const now = flowUnder(await matrixOf(page), origin, at);
		expect(Math.abs(now.x - held.x), 'the tapped point stayed under the finger, horizontally').toBeLessThan(4);
		expect(Math.abs(now.y - held.y), 'the tapped point stayed under the finger, vertically').toBeLessThan(4);
	});

	test('and dragging the other way zooms back out', async ({ page }) => {
		const box = (await pane(page).boundingBox())!;
		const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height * 0.3) };
		const before = await scaleOf(page);

		const touch = await touchSession(page);
		await touch.down(at);
		await touch.up();
		await touch.down(at);
		await dragBy(page, touch, at, Math.min(200, Math.round(box.height * 0.4)));
		await touch.up();

		await expect
			.poll(() => scaleOf(page), { message: 'dragging down zoomed out' })
			.toBeLessThan(before * 0.7);
	});

	test('a single tap and drag still PANS, and does not zoom', async ({ page }) => {
		// The regression this gesture could plausibly cause, and the one nothing else would catch.
		const at = await emptySpot(page);
		const before = await matrixOf(page);

		const touch = await touchSession(page);
		await touch.down(at);
		await dragBy(page, touch, at, 120);
		await touch.up();

		await expect
			.poll(() => matrixOf(page).then((m) => m.y - before.y), { message: 'one finger panned' })
			.toBeGreaterThan(60);
		expect(await scaleOf(page), 'and panning is not zooming').toBeCloseTo(before.k, 5);
	});
});

test.describe('binding a node to a panel by drag', () => {
	/**
	 * R Task 5 (§3.1h) — dragging a node onto a panel is the ONLY way to populate a Viewer /
	 * Parameters / Metadata / Console panel (`NodeLinkedPanel` literally says "Drag a node here"), and
	 * it never worked under touch: `linkTargetAt` read `event.clientX`, which a `TouchEvent` does not
	 * have, so it returned null on every touch drag and `ws.linkNodeToPanel` was unreachable.
	 * `editor/eventPoint.ts` owns the extraction now (unit-tested); this is the wiring proof.
	 */

	interface Pt {
		x: number;
		y: number;
	}

	async function touchSession(page: Page) {
		const cdp = await page.context().newCDPSession(page);
		const send = (type: string, touchPoints: Pt[]) =>
			cdp.send('Input.dispatchTouchEvent', { type, touchPoints } as never);
		return {
			down: (p: Pt) => send('touchStart', [p]),
			moveTo: (p: Pt) => send('touchMove', [p]),
			up: (p: Pt) => send('touchEnd', [p])
		};
	}

	const panels = (page: Page) => page.locator('[data-panel-id]');

	test('a node dragged onto a viewer panel binds it, under touch', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const before = await panels(page).count();
		const touch = await touchSession(page);

		await page.getByTestId('panel-header').first().click({ button: 'right' });
		await page.locator('.context-menu .item', { hasText: 'Split Right' }).first().click();
		await expect(panels(page)).toHaveCount(before + 1);

		const uid = await addNode(page, 'Oscillator', 'inputs', [10, 10]);
		try {
			await waitForNode(page, uid);
			const target = panels(page).nth(1);
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'viewer'),
				await target.getAttribute('data-panel-id')
			);

			const head = (await page.locator(`.goofi-node .header`).first().boundingBox())!;
			const dst = (await target.boundingBox())!;
			const from = { x: Math.round(head.x + head.width / 2), y: Math.round(head.y + head.height / 2) };
			const to = { x: Math.round(dst.x + dst.width / 2), y: Math.round(dst.y + dst.height / 2) };

			await touch.down(from);
			// Two steps: the first is what SvelteFlow reads as "this is a drag", the second lands it.
			await touch.moveTo({ x: from.x + 12, y: from.y + 12 });
			await touch.moveTo(to);
			await touch.up(to);

			// The bind is a `set_panel` command, so it shows up when the manager's delta does.
			await expect
				.poll(
					() =>
						page.evaluate(
							() =>
								(
									(window as any).goofi.query.panels() as Array<{ type: string; node: string | null }>
								).find((p) => p.type === 'viewer')?.node ?? null
						),
					{ message: 'the drop bound the node to the viewer panel' }
				)
				.toBe(uid);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
			await page
				.getByTestId('panel-header')
				.nth(1)
				.getByRole('button', { name: 'Close panel' })
				.click();
			await expect(panels(page)).toHaveCount(before);
		}
	});
});

test.describe('what an input slot names, and when', () => {
	/**
	 * The input-name tag under a coarse pointer, after the always-open rule was replaced.
	 *
	 * `.conn-label` is the ONLY rendering of an input slot's name. A mouse hovers the pill to ask for
	 * it; a finger cannot, so R rested every one of them open under the coarse idiom — which put a name
	 * tag on every input on the whole canvas, permanently, for a question nobody was asking. The door is
	 * a PROXIMITY reveal now, scoped to a cable actually being in flight, and it is the same door on
	 * both modalities rather than a phone-only rule.
	 *
	 * Both halves are pinned here because both can regress independently: the tag resting hidden (which
	 * the deleted rule would restore) and the tag appearing for the input the cable is closing on but
	 * not for a distant one (which a missing or oversized radius would break).
	 */

	test('with no cable in flight, an input names nothing', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const stage = await proximityStage(page);
		try {
			await expect(
				inputLabel(page, stage.near),
				'the always-open coarse rule is gone — a name tag is an answer to a question'
			).toHaveCSS('opacity', '0');
			await expect(inputLabel(page, stage.far)).toHaveCSS('opacity', '0');
		} finally {
			await clearStage(page, stage);
		}
	});

	test('a cable in flight names the input it is closing on, and not a distant one', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const stage = await proximityStage(page);
		const touch = await touchSession(page);
		let down = false;
		try {
			const from = await outHandle(page, stage.src);
			const to = await inConn(page, stage.near);

			await touch.down(from);
			down = true;
			// Walk the cable across, ending ON the target input. Stepped, and with the finger let come to
			// rest before anything is read: Chromium holds the first touchmoves back behind its touch slop,
			// and reads back-to-back synthetic moves as a fling.
			for (const f of [0.3, 0.6, 0.85, 1]) {
				await touch.moveTo({
					x: Math.round(from.x + (to.x - from.x) * f),
					y: Math.round(from.y + (to.y - from.y) * f)
				});
				await page.waitForTimeout(30);
			}
			await page.waitForTimeout(180);

			await expect(
				inputLabel(page, stage.near),
				'the input the cable is closing on names itself'
			).toHaveCSS('opacity', '1');
			await expect(
				inputLabel(page, stage.far),
				'…and one a screenful away stays quiet — a reveal, not a floodlight'
			).toHaveCSS('opacity', '0');

			// Let go over bare canvas, so this measures the reveal and never wires anything.
			const away = await bareSpot(page);
			await touch.moveTo(away);
			await page.waitForTimeout(180);
			await touch.up();
			down = false;
			await expect(
				inputLabel(page, stage.near),
				'and the tag goes away with the cable that summoned it'
			).toHaveCSS('opacity', '0');
		} finally {
			if (down) await touch.up();
			await clearStage(page, stage);
		}
	});
});

test.describe('a viewer on the canvas', () => {
	/**
	 * The in-canvas slot viewer under a coarse pointer.
	 *
	 * A node's viewer chrome was built for a mouse: five `:hover` rules tint the header, brighten the
	 * triangle, wash the slot name and light the kind select and the cog. None of them can ever fire on
	 * a phone — but the browser still MATCHES `:hover` for a synthetic pointer, so on a hybrid the wrong
	 * half of the app lights up, and on a phone they are dead weight in the stylesheet. They are gated
	 * on `(hover: hover)` now: a device that hovers gets all five, a device that does not gets none.
	 *
	 * The other half is that the viewer is a large part of a node's surface, and a finger that lands on
	 * it is reaching for the NODE. `SlotViewer`'s header swallowed its own `pointerdown` to "keep
	 * SvelteFlow from starting a node drag"; that claim went stale when `@xyflow/svelte` moved node
	 * dragging onto d3-drag (`mousedown`/`touchstart`), so the swallow reached nothing — the second test
	 * here is what keeps it that way now that touch releases it outright.
	 *
	 * Driven through CDP touch, not `page.mouse`: under `hasTouch` Playwright's mouse API still
	 * dispatches MOUSE events. The hover test is the exception and says so where it moves the mouse —
	 * the point there is that even a real hover must not paint under this media query.
	 */

	const slot = (page: Page, uid: string) => page.locator(`.slot-viewer[data-node="${uid}"]`);

	async function oscillator(page: Page, pos: [number, number]): Promise<string> {
		const uid = await addNode(page, 'Oscillator', 'inputs', pos);
		await waitForNode(page, uid);
		await expect(slot(page, uid)).toBeVisible();
		return uid;
	}

	async function remove(page: Page, uid: string): Promise<void> {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}

	const nodePos = (page: Page, uid: string): Promise<[number, number]> =>
		page.evaluate(
			(u) => (window as any).goofi.query.graph().nodes.find((n: any) => n.uid === u)?.pos,
			uid
		);

	/** The canvas scale, so a screen-space drag can be compared against a flow-space position. */
	const zoomOf = (page: Page): Promise<number> =>
		page
			.locator('.svelte-flow__viewport')
			.first()
			.evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).a);

	const centreOf = async (page: Page, sel: string): Promise<TouchPoint> => {
		const box = await page.locator(sel).first().boundingBox();
		expect(box, `${sel} is on screen`).toBeTruthy();
		return { x: Math.round(box!.x + box!.width / 2), y: Math.round(box!.y + box!.height / 2) };
	};

	test('the slot viewer paints no hover feedback where there is no hover', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await oscillator(page, [40, 40]);
		try {
			// Every resting value first, with the pointer parked off the node — reading one AFTER
			// hovering another would bake a hover state into the baseline (the slot name and the cog both
			// sit inside the header, whose own tint is one of the five).
			await unhover(page);
			const rest = new Map<string, string[]>();
			for (const s of VIEWER_HOVER_SURFACES) rest.set(s.name, await surfaceStyles(page, uid, s));

			for (const s of VIEWER_HOVER_SURFACES) {
				await hoverSettled(page, uid, s);
				const el = page.locator(`.slot-viewer[data-node="${uid}"] ${s.sel}`).first();
				for (const [i, prop] of s.props.entries()) {
					await expect(el, `${s.name} keeps its resting ${prop} under a coarse pointer`).toHaveCSS(
						prop,
						rest.get(s.name)![i]
					);
				}
				await unhover(page);
			}
		} finally {
			await remove(page, uid);
		}
	});

	test('a touch drag that starts on the viewer moves the node, like one on its body', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await oscillator(page, [40, 40]);
		try {
			const touch = await touchSession(page);
			const nodeSel = `.svelte-flow__node[data-id="${uid}"]`;

			/** Drag `sel`'s centre by (dx, dy) screen px and answer the node's COMMITTED displacement in
			 *  screen px (the flow-space move scaled by the live zoom, so it is the same ruler the finger
			 *  travelled in). Reading the graph store, not the box, so this measures the move that
			 *  actually landed in the patch. */
			const dragFrom = async (sel: string, dx: number, dy: number): Promise<[number, number]> => {
				const at = await centreOf(page, sel);
				const zoom = await zoomOf(page);
				const before = await nodePos(page, uid);
				await touch.down(at);
				for (const f of [0.2, 0.45, 0.7, 1]) {
					await touch.moveTo({ x: Math.round(at.x + dx * f), y: Math.round(at.y + dy * f) });
					await page.waitForTimeout(25);
				}
				// Chromium reads back-to-back synthetic moves as a FLING, which eats the next tap anywhere
				// on the page. Let the finger come to rest before lifting.
				await page.waitForTimeout(180);
				await touch.up();
				await page.waitForTimeout(500);
				const after = await nodePos(page, uid);
				return [(after[0] - before[0]) * zoom, (after[1] - before[1]) * zoom];
			};

			/**
			 * Measured against the node's OWN body, not against the finger's 60px.
			 *
			 * A touch drag always lags its finger by Chromium's touch slop — the platform holds the first
			 * `touchmove`s back until the contact has travelled far enough to be a drag rather than a tap,
			 * and the gesture then tracks 1:1 from wherever that was. Measured here at ~27px, and it is
			 * the platform's, not the app's (`nodeDragThreshold` is 1). It applies to every drag equally,
			 * so the honest question — and the one the requirement asks — is whether a drag that starts on
			 * the viewer moves the node exactly as far as one that starts on its body.
			 */
			const ref = await dragFrom(`${nodeSel} .header`, 60, 30);
			expect(ref[0], 'the reference drag really moved the node').toBeGreaterThan(20);

			for (const [what, sel] of [
				['PLOT', `.slot-viewer[data-node="${uid}"] .body`],
				// The header bar is the half that swallowed its own `pointerdown` to keep the drag off.
				['HEADER', `.slot-viewer[data-node="${uid}"] header`]
			] as const) {
				const got = await dragFrom(sel, 60, 30);
				expect(got[0], `a drag from the viewer ${what} moves the node as its body does (x)`).toBeCloseTo(
					ref[0],
					0
				);
				expect(got[1], `…and (y)`).toBeCloseTo(ref[1], 0);
			}
		} finally {
			await remove(page, uid);
		}
	});

	test('a tap is not a drag — the cog, the kind picker and collapse still answer', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await oscillator(page, [40, 40]);
		try {
			const touch = await touchSession(page);
			const menu = page.getByTestId('viewer-settings-menu');
			const tap = async (sel: string): Promise<void> => {
				await touch.down(await centreOf(page, sel));
				await touch.up();
				await page.waitForTimeout(150);
			};
			const sv = `.slot-viewer[data-node="${uid}"]`;

			await tap(`${sv} [data-testid="viewer-settings-cog"]`);
			await expect(menu, 'the cog opens its settings on a tap').toBeVisible();
			await page.keyboard.press('Escape');
			await expect(menu).toBeHidden();

			await tap(`${sv} [data-testid="viewer-kind"] select`);
			expect(
				await page.evaluate(
					() => document.activeElement?.closest('[data-testid]')?.getAttribute('data-testid') ?? ''
				),
				'the kind picker takes the tap'
			).toBe('viewer-kind');

			// The disclosure triangle, which is the collapse control a keyboard user gets too.
			await tap(`${sv} .tri`);
			await expect(slot(page, uid), 'the triangle collapses the viewer on a tap').toHaveClass(
				/collapsed/
			);
			await tap(`${sv} header`);
			await expect(slot(page, uid), 'and the header bar expands it again').not.toHaveClass(
				/collapsed/
			);
		} finally {
			await remove(page, uid);
		}
	});
});

test.describe('the panel header\u2019s long-press menu', () => {
	/**
	 * The panel header's coarse-pointer door (D-R5).
	 *
	 * Every structural action a panel has — Split Right, Split Down, Maximize, Change content, Close —
	 * lives in one context menu opened by `oncontextmenu`. The header is `role="toolbar"
	 * tabindex="-1"` with no keydown handler, so on a phone Split Right and Split Down had **no door
	 * at all**. A long press is the door; the desktop right-click is untouched.
	 *
	 * And once the menu is open, *"Change content ▸"* still could not be expanded: its submenus were
	 * `mouseenter`-only and clicking a parent row was an explicit no-op. That is part of this item.
	 *
	 * `maximize` is what the action half is proved with on purpose: `maximizedPanelId` is a store
	 * field outside the arrangement, so it provably cannot reach the manager or the `.gfi` and this
	 * spec cannot leave a layout behind for a later one (a worker's specs share one backend).
	 */

	const HOLD_MS = 700; // the recognizer fires at 500

	function header(page: Page): Locator {
		return page.getByTestId('panel-header').first();
	}

	function menuRow(page: Page, label: string): Locator {
		return page
			.locator('.context-menu .item')
			.filter({ has: page.locator('.label', { hasText: new RegExp(`^${label}$`) }) });
	}

	/** Long-press the panel header and return the panel id it belongs to. */
	async function pressHeader(page: Page): Promise<string> {
		const hdr = header(page);
		const panelId = (await hdr.evaluate(
			(el) => el.closest('.panel')!.getAttribute('data-panel-id')!
		)) as string;
		const box = (await hdr.boundingBox())!;
		// Press on the header's empty middle — not on the content button or the icon buttons, which
		// have their own actions and must not double as a menu trigger.
		const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height / 2) };
		const touch = await touchSession(page);
		await touch.down(at);
		await page.waitForTimeout(HOLD_MS);
		await touch.up();
		return panelId;
	}

	test('a long press on the panel header opens the structural menu', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await pressHeader(page);

		await expect(page.locator('.context-menu').first(), 'the press opened a menu').toBeVisible();
		for (const label of ['Split Right', 'Split Down', 'Maximize', 'Change content', 'Close Panel']) {
			await expect(menuRow(page, label), label).toBeVisible();
		}
		await page.keyboard.press('Escape');
		await expect(page.locator('.context-menu')).toHaveCount(0);
	});

	test('the press leaves the panel active — it does not swallow setActive', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId = await pressHeader(page);
		// `Panel.svelte` marks the active panel on a CAPTURE-phase pointerdown. A recognizer that
		// stopped propagation would freeze `activePanelId` and quietly drift selection scoping.
		await expect(page.locator(`.panel.active[data-panel-id="${panelId}"]`)).toHaveCount(1);
		await page.keyboard.press('Escape');
	});

	test('a menu row acts: Maximize from the long-press menu maximizes the panel', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await pressHeader(page);
		await menuRow(page, 'Maximize').tap();
		await expect(page.locator('.context-menu')).toHaveCount(0);
		// The header's own control flips to Restore — the direct observable of `maximizedPanelId`.
		const restore = header(page).getByRole('button', { name: 'Restore panel' });
		await expect(restore, 'the panel is maximized').toBeVisible();

		// …and back, so the panel is handed on as it was found.
		await restore.tap();
		await expect(header(page).getByRole('button', { name: 'Maximize panel' })).toBeVisible();
	});

	test('“Change content ▸” expands on tap, and picking a type switches the panel', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId = await pressHeader(page);

		const parent = menuRow(page, 'Change content');
		await parent.tap();
		// The submenu is a second `.context-menu` portalled beside the first.
		await expect(page.locator('.context-menu'), 'the submenu opened on a tap').toHaveCount(2);

		await menuRow(page, 'Console').tap();
		await expect(page.locator('.context-menu')).toHaveCount(0);
		await expect(page.locator(`.panel[data-panel-id="${panelId}"]`)).toHaveAttribute(
			'data-panel-type',
			'console'
		);

		// Hand the panel back to the node editor. The type change is a command, so the canvas coming
		// back is proof the manager already holds the restore.
		await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'), panelId);
		await expect(page.locator('.canvas-wrap').first()).toBeVisible();
	});
});

test.describe('splitting without a corner grip', () => {
	/**
	 * Splitting a panel, from a finger.
	 *
	 * The Blender-style corner grip — drag a panel corner inward to split, or onto a sibling to join —
	 * is a FINE-POINTER gesture and always has been: `Panel.svelte` deliberately gives the grip no
	 * `touch-action`, so a touch that lands on one and moves is reclaimed by the browser as a pan.
	 * A 16px triangle that is invisible, un-draggable and still hit-testable is the worst of the three
	 * — it eats a tap in the one corner the editor's zoom cluster sits in — so on touch the grips are
	 * taken off the board outright and the door is the panel header instead.
	 *
	 * `panel-corner-split.spec.ts` is this file's fine-pointer twin: it proves the same grips still
	 * arm, preview and commit a split under a mouse. The pair is what makes "touch only" a measured
	 * claim rather than a media query nobody reads.
	 *
	 * And the interlock at the bottom is the point of taking the grips away at all: the panel header
	 * carries Split Right and Split Down as real controls now (progressive overflow, `overflowFit.ts`),
	 * so a finger that lost the corner gesture did not lose the operation. That test asserts against
	 * the WORKSPACE TREE (`goofi.query.panels()`), not the DOM, because gaining a `.panel` element is
	 * not the same claim as gaining a panel.
	 */

	type Corner = 'tl' | 'tr' | 'bl' | 'br';
	const CORNERS: Corner[] = ['tl', 'tr', 'bl', 'br'];

	/** The point 3px inside a panel body's corner — inside the grip's 16px clipped triangle. */
	async function cornerPoint(page: Page, corner: Corner): Promise<{ x: number; y: number }> {
		const body = (await page.locator('.panel-body').first().boundingBox())!;
		return {
			x: Math.round(corner[1] === 'l' ? body.x + 3 : body.x + body.width - 3),
			y: Math.round(corner[0] === 't' ? body.y + 3 : body.y + body.height - 3)
		};
	}

	test('the corner split grips are not rendered under a coarse pointer', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		for (const c of CORNERS) {
			await expect(
				page.locator(`.panel-body .corner.${c}`).first(),
				`the ${c} grip is off the board on touch`
			).toBeHidden();
		}
	});

	/* Hidden is only half of it: an `opacity: 0` box still hit-tests, and this one carries
	   `z-index: var(--z-chrome)` over the panel's own content. So the affordance must be gone from the
	   hit test too, not merely from the paint. */
	test('nothing in a panel corner hit-tests as a grip on touch', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		for (const c of CORNERS) {
			const at = await cornerPoint(page, c);
			const hit = await page.evaluate(
				(p) => (document.elementFromPoint(p.x, p.y) as HTMLElement | null)?.className ?? '',
				at
			);
			expect(hit, `the ${c} corner belongs to the panel content, not to a grip`).not.toContain(
				'corner'
			);
		}
	});

	test('a touch drag out of a panel corner splits nothing', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panels = page.locator('.panel');
		const before = await panels.count();

		const at = await cornerPoint(page, 'tl');
		const touch = await touchSession(page);
		await touch.down(at);
		// Well past THRESHOLD (24px), the distance that arms a split intent and paints its ghost.
		await touch.moveTo({ x: at.x + 100, y: at.y + 12 });
		await touch.moveTo({ x: at.x + 180, y: at.y + 16 });
		// Let the finger rest: back-to-back synthetic moves read as a fling, which eats the next tap.
		await page.waitForTimeout(150);
		await expect(page.locator('.drag-ghost'), 'no split is being previewed').toHaveCount(0);
		await touch.up();
		await page.waitForTimeout(200);

		expect(await panels.count(), 'the workspace is untouched').toBe(before);
	});

	/* ---------------------------------------------------------------------------------------------
	   The interlock: with the corner gesture gone, the header is what a finger splits with.
	   -------------------------------------------------------------------------------------------- */

	const hdr = (page: Page, n = 0): Locator => page.getByTestId('panel-header').nth(n);

	/** How many panels the WORKSPACE TREE holds — the claim, as opposed to how many `.panel` elements
	 *  happen to be mounted. */
	const treePanels = (page: Page): Promise<number> =>
		page.evaluate(() => (window as any).goofi.query.panels().length as number);

	/** Tap Split Down wherever the header is currently keeping it: inline while it fits, and behind the
	 *  ⋯ once the panel is too narrow for it. Both are the same command (D-R2), and a phone will meet
	 *  whichever the width decides — so the door the test uses is the one the header is offering. */
	async function splitDownFromHeader(page: Page): Promise<void> {
		const inline = hdr(page).getByTestId('panel-split-column');
		if (await inline.isVisible()) {
			await inline.tap();
			return;
		}
		await hdr(page).getByTestId('panel-overflow').tap();
		await expect(page.locator('.context-menu').first()).toBeVisible();
		await page
			.locator('.context-menu .item')
			.filter({ has: page.locator('.label', { hasText: /^Split Down$/ }) })
			.tap();
	}

	test('a finger can still split a panel — the header carries what the corner gave up', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		expect(await treePanels(page), 'the workspace starts as one panel').toBe(1);

		try {
			await splitDownFromHeader(page);
			await expect
				.poll(() => treePanels(page), { message: 'the workspace tree really gained a panel' })
				.toBe(2);
			await expect(page.locator('.panel')).toHaveCount(2);
		} finally {
			await hdr(page, 1).getByRole('button', { name: 'Close panel' }).tap();
			await expect(page.locator('.panel'), 'the workspace is handed back').toHaveCount(1);
		}
	});
});

test.describe('the panel toolbar', () => {
	/* The coarse half of `panel-bar.spec.ts`, and the half that has to be measured rather than
	   reasoned about: under a coarse pointer `--panel-header-h` IS `--hit`, so a 44px bar has exactly
	   room for its 44px controls and nothing else. Every control in it that carries the tap floor as a
	   BORDER box fits; one that carries it as a content box (a <select> did, at 44 + 2px of border)
	   pushes the whole strip past the header. `touch-hit-floor.spec.ts` measures the other direction —
	   that the floor survived the shortening. */
	test('every panel’s toolbar is exactly as tall as the panel header above it', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await barsMatchTheHeader(page);
	});

	/* The coarse half of the one-gap rule, and not a re-measure of a constant: under a coarse pointer
	   the viewer bar's controls (215px) no longer fit the slack they are given (184px), so the strip's
	   control host is a live horizontal scroller here and nowhere else. The gaps have to survive that. */
	test('a panel’s toolbar spaces every control it holds by the strip’s own gap', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await controlsSitAtOneGap(page);
	});
});

test.describe('the inspector as a bottom sheet', () => {
	/**
	 * The inspector as a bottom sheet — the `touch` (Pixel 7 portrait) project, where the host editor
	 * panel is decisively taller than it is wide and `@container (orientation: portrait)` fires.
	 *
	 * This is not a touch-only MODE (the governing principle of the spec): nothing asserted here is
	 * gated on the pointer, and `inspector-orientation.spec.ts` proves the very same sheet on a mouse at
	 * a landscape desktop window. What this file adds is the geometry only a phone has — 412×915, where
	 * a right-anchored pane left a 44px strip of canvas — plus the soft keyboard, which no desktop
	 * project can fake honestly. The gesture itself is `touch-modality.spec.ts`'s, in both anchors.
	 *
	 * The pane slides over `--dur-slow`, so every position here is read past the app's own
	 * reduced-motion collapse and through a settling read; a one-shot measurement mid-slide describes a
	 * frame of the animation, not the layout.
	 */

	/** The soft-keyboard overlap the app published, in px. */
	const kbInset = (page: Page): Promise<number> =>
		page.evaluate(
			() => parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--kb-inset')) || 0
		);

	test('the pane rises from the bottom and rests at 60% of its host (D-I6)', async ({ page }) => {
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			// The axis is a CSS fact, and the one the drag handler reads back — so read it the same way
			// rather than inferring it from a box that has already been laid out.
			await expect
				.poll(() => paneAxis(page), { message: 'the sheet slid up, not in from the right' })
				.toBe('y');
			const host = (await editor(page).boundingBox())!;
			const p = await settledBox(pane(page));
			expect(p.width, 'it spans the host').toBeCloseTo(host.width, 0);
			expect(p.x, 'from its left edge').toBeCloseTo(host.x, 0);
			expect(host.y + host.height - (p.y + p.height), 'flush with the bottom').toBeLessThan(2);
			expect(p.height, 'and 60% of it tall').toBeCloseTo(host.height * 0.6, 0);
		} finally {
			await drop(page, uid);
		}
	});

	test('the canvas above the sheet still takes a tap that changes the selection', async ({ page }) => {
		// Success criterion §4.1, and the whole reason the pane is capped at all: before R the pane's only
		// host clamp reserved one `--hit`, which on a 412px phone is a 44px strip — and a strip is the
		// only way out, since deselecting means tapping canvas the pane covers.
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			const p = await settledBox(pane(page));
			// `emptySpot` only resolves where the flow pane is really the TOPMOST element, so a point it
			// returns is by construction neither under the sheet nor under its grip band — and the 40% of
			// host it has to find one in is exactly the canvas the 60% cap hands back. It also clears
			// every node card by a tap target, because Chromium's touch adjustment would otherwise snap
			// a tap taken beside a card ONTO it and re-select the node this tap means to drop.
			const spot = await emptySpot(page);
			expect(spot.y, 'the live canvas it found is above the sheet').toBeLessThan(p.y);
			await page.touchscreen.tap(spot.x, spot.y);
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length), {
					message: 'the tap reached the canvas and changed the selection'
				})
				.toBe(0);
		} finally {
			await drop(page, uid);
		}
	});

	/* The edge drag used to be re-stated here, in the Y axis, beside its X-axis twin in
	   `inspector-orientation.spec.ts`. It is `touch-modality.spec.ts`'s now: that file runs one copy,
	   normalised to the anchored axis, in BOTH the `touch` and `touch-landscape` projects — which is
	   the only arrangement that can catch the two drifting apart. Two copies each asserting its own
	   orientation is precisely how the resting grabber came to be orientation-gated in the first
	   place. */

	test('the soft keyboard lifts the sheet off the bottom (D-I7)', async ({ page }) => {
		// `--kb-inset` is the one thing R kept of the device seam, precisely because CSS cannot see the
		// soft keyboard — and a text field inside a bottom sheet is the case that needed it. This is its
		// fourth consumer.
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			const host = (await editor(page).boundingBox())!;
			const down = await settledBox(pane(page));
			expect(host.y + host.height - (down.y + down.height), 'resting on the bottom').toBeLessThan(2);

			const KB = 260;
			await setKeyboardInset(page, KB);
			await expect.poll(() => kbInset(page), { message: 'the app measured the keyboard' }).toBe(KB);
			const up = await settledBox(pane(page));
			expect(
				host.y + host.height - (up.y + up.height),
				'the sheet sits a keyboard above the bottom'
			).toBeCloseTo(KB, 0);
			// …and it did not merely grow upward past its cap to do it.
			expect(up.height, 'still 60% tall').toBeCloseTo(down.height, 0);
		} finally {
			await setKeyboardInset(page, 0);
			await drop(page, uid);
		}
	});
});

test.describe('the expression editor under a finger', () => {
	/**
	 * The coarse half of the expression editor (D-X8). Runs ONLY under the `touch` project (Pixel 7 →
	 * `(pointer: coarse)` / `(hover: none)` true, real touch input available), because every claim here is
	 * about a device the `default` project is not.
	 *
	 * Three things touch needs that a desktop editor gets for free, and none of them are inherited: the
	 * editable element is a contenteditable `<div>`, so `app.css`'s `input, select, textarea` coarse floors
	 * — both the 16px focus-zoom one and the `--hit` one — cannot reach it; and the completion popup is
	 * placed by CodeMirror, whose own default measures `window.innerHeight`, which a soft keyboard does not
	 * shrink.
	 *
	 * Driven against a real node, through the doors a phone has: place an Oscillator, tap it, tap fx
	 * on. The four names the evaluator injects (`nd`, `t`, `np`, `globals`) are offered with no patch
	 * content at all, through the same merged popup as Python's own builtins.
	 */

	const MARGIN = 6; // the popup's viewport-edge margin, matching clampToViewport's

	// The helper below leaves a node on the shared backend, so hand the patch back per test rather
	// than per call — a second Oscillator would put a second card under the tap.
	test.afterEach(async ({ page }) => {
		const uids: string[] = await page.evaluate(() =>
			(window as any).goofi.query.graph().nodes.map((n: { uid: string }) => n.uid)
		);
		if (uids.length === 0) return;
		await page.evaluate((us) => (window as any).goofi.commands.removeNodes(us), uids);
		await expect.poll(() => page.evaluate(() => (window as any).goofi.query.graph().nodes.length)).toBe(0);
	});

	/** An Oscillator's `amplitude`, switched into expression mode; returns its inline editor. */
	async function fxEditor(page: Page): Promise<Locator> {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		await tapNode(page, uid);
		const field = page.getByTestId('auto-side-panel').getByTestId('param-field-amplitude');
		await field.getByTestId('param-fx-toggle').tap();
		const editor = field.getByTestId('param-expr-input');
		await expect(editor).toBeVisible();
		return editor;
	}

	const popup = (page: Page) => page.locator('.cm-tooltip-autocomplete');

	/** Type into the editor from scratch and settle past the completion debounce. */
	async function retype(page: Page, editor: Locator, src: string): Promise<void> {
		await editor.tap();
		await page.keyboard.press('Control+a');
		await page.keyboard.press('Delete');
		await page.keyboard.type(src, { delay: 10 });
		await page.waitForTimeout(300);
	}

	test('the inline expression editor is >= 16px under a coarse pointer', async ({ page }) => {
		const editor = await fxEditor(page);
		// `.cm-content` inherits from the host, which is where the floor is stated — so this measures the
		// element that actually takes focus, the same way `touch-inspector.spec.ts` measures the expanded one.
		const px = await editor.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
		expect(px, 'below 16px iOS force-zooms the viewport on focus').toBeGreaterThanOrEqual(16);
	});

	test('a completion accepts by TAP, and its rows are real tap targets', async ({ page }) => {
		const editor = await fxEditor(page);
		await retype(page, editor, 'n');
		await expect(popup(page), 'the merged popup opened').toHaveCount(1);
		const row = popup(page).locator('li', { hasText: 'nd' }).first();
		await expect(row).toBeVisible();
		const box = (await row.boundingBox())!;
		const hit = await page.evaluate(() =>
			parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hit'))
		);
		expect(box.height, 'a completion row is as tappable as anything else').toBeGreaterThanOrEqual(hit);

		await row.tap();
		await expect(editor, 'the tap accepted the completion').toHaveText('nd');
	});

	/**
	 * D-X8's popup clause. The tooltip is parented to `document.body` and constrained through CodeMirror's
	 * `tooltipSpace` hook by `overlayViewport()` — the app's one soft-keyboard-aware measurement, the same
	 * one `Popover` and `ContextMenu` clamp against. CodeMirror's own default reads `window.innerHeight`,
	 * which a keyboard does not shrink, so without the hook the list is parked underneath the very keyboard
	 * that raised it.
	 *
	 * The inset is derived from the popup's own measured box, so the test tunes itself: enough to push its
	 * bottom edge under the keyboard, not so much that it no longer fits above it.
	 */
	test('the completion popup stays clear of the soft keyboard', async ({ page }) => {
		const editor = await fxEditor(page);
		await retype(page, editor, 'n');
		await expect(popup(page)).toBeVisible();
		const before = (await popup(page).boundingBox())!;
		const innerHeight = await page.evaluate(() => window.innerHeight);
		const inset = innerHeight - (before.y + before.height) + 20;
		expect(inset, 'the popup starts clear of the bottom edge, so an inset can reach it').toBeGreaterThan(0);

		try {
			// The keyboard is up BEFORE the list opens, which is the real order: you type with it showing.
			await setKeyboardInset(page, inset);
			await retype(page, editor, 'np');
			await expect(popup(page)).toBeVisible();
			const box = (await popup(page).boundingBox())!;
			expect(box.y, 'the popup stays on-screen').toBeGreaterThanOrEqual(MARGIN - 0.5);
			expect(
				box.y + box.height,
				'the popup sits above the keyboard, not underneath it'
			).toBeLessThanOrEqual(innerHeight - inset - MARGIN + 0.5);
		} finally {
			await setKeyboardInset(page, 0);
		}
	});
});

test.describe('the narrowest phone', () => {
	/**
	 * R Task 5 (§3.1e/f) — what a narrow viewport clips today.
	 *
	 * Each of these surfaces has a fixed cost it never gives up, so the thing at the END of the row is
	 * the thing that disappears: the file browser's Save button, the console's copy button, the globals
	 * table's inputs. "Off the right edge with no scroll recovery" is not a small-screen inconvenience,
	 * it is the control not existing.
	 *
	 * Measured against the CONTAINER, never a literal, so the assertions still mean something at any
	 * width and cannot be greened by a lucky viewport.
	 *
	 * The two checks that are about the SHAPE of the viewport rather than a row's fixed cost — the
	 * inspector's clamp against its host and the add-node menu's clamp against the screen — moved to
	 * `touch-reflow.spec.ts`, which three projects run at three real device geometries.
	 */

	/** How far `inner` overflows `outer` horizontally, in px (0 = fully inside). */
	async function overflowRight(inner: Locator, outer: Locator): Promise<number> {
		const i = (await inner.boundingBox())!;
		const o = (await outer.boundingBox())!;
		expect(i, 'the control is rendered at all').toBeTruthy();
		return Math.max(0, i.x + i.width - (o.x + o.width));
	}

	/** Borrow the first panel for `type`, run `body`, and hand the workspace back. */
	async function withPanelType(page: Page, type: string, body: () => Promise<void>): Promise<void> {
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		await page.evaluate(
			([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
			[panelId, type] as const
		);
		try {
			await body();
		} finally {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			// The restore is a command; the assertion above proves the manager already applied it.
		}
	}

	/* 320px — the narrowest phone still in real use, and the width at which every fixed cost in a row
	   stops fitting. The Pixel-7 project's own 412px is where these surfaces got tight; this is where
	   they broke. `hasTouch`/`isMobile` carry over, so the coarse floors are still in force. */
	test.describe('at 320px', () => {
		test.use({ viewport: { width: 320, height: 690 } });

		test('the file browser keeps its Save button on screen', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			// Ctrl+S, not the TopBar button: the header's own actions are still off-screen here (that is
			// Task 6's bug, not this one). A fresh page has no save path, so Save opens the browser.
			await page.keyboard.press('Control+s');
			const modal = page.getByTestId('fs-browser');
			await expect(modal).toBeVisible();
			try {
				expect(
					await overflowRight(page.getByTestId('fs-save'), modal),
					'the Save button is inside the modal'
				).toBe(0);
				expect(
					await overflowRight(page.getByTestId('fs-filename'), modal),
					'and so is the name it saves under'
				).toBe(0);

				// The root sidebar was non-shrinking, so the file list got whatever was left of a modal
				// that is itself only 92vw. Compared against the modal, so this holds at any width.
				const list = (await page.getByTestId('fs-list').boundingBox())!;
				const m = (await modal.boundingBox())!;
				expect(list.width, 'the file list gets the majority of a narrow modal').toBeGreaterThan(
					m.width * 0.6
				);
			} finally {
				await page.keyboard.press('Escape');
				await expect(modal).toHaveCount(0);
			}
		});

		test('the globals table keeps its inputs usable and its delete reachable', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const name = `narrow_${Date.now()}`;
			await addGlobal(page, name, 1, 'float');
			await withPanelType(page, 'globals', async () => {
				try {
					const row = page.locator(`[data-testid="global-row"][data-name="${name}"]`);
					await expect(row).toBeVisible();
					expect(
						await overflowRight(row.getByTestId('global-delete'), row),
						'the delete control is inside its row'
					).toBe(0);

					const r = (await row.boundingBox())!;
					const nameBox = (await row.getByTestId('global-name').boundingBox())!;
					// The actions column claimed a flat 15% but never rendered under ~85px, so it stole
					// from the two editable columns at every width below ~565px.
					expect(
						nameBox.width,
						'the name is still editable, not squeezed to a sliver'
					).toBeGreaterThan(r.width * 0.3);
				} finally {
					await page.evaluate((n) => (window as any).goofi.commands.removeGlobal(n), name);
				}
			});
		});

		test('a console row keeps its copy button inside the scroller', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			await withPanelType(page, 'console', async () => {
				// A node erroring on its empty required input (see `addErroringNode`) — one console row.
				const uid = await addErroringNode(page);
				try {
					await expect(page.getByTestId('console-entry').first()).toBeVisible();
					// `.scroll` clips horizontally, so anything past its right edge is simply gone — the
					// row's ~258px of fixed cost is what pushed the copy button out.
					expect(
						await overflowRight(page.getByTestId('console-copy').first(), page.locator('.scroll')),
						'the copy button is inside the scroller that clips it'
					).toBe(0);
				} finally {
					await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
					await waitForNoNode(page, uid).catch(() => {});
				}
			});
		});
	});
});

test.describe('the whole authoring pass, on a phone', () => {
	/**
	 * **The headline test for sub-project R.** Its §5 success criterion, verbatim: at 412px portrait a
	 * user can *add a node, connect it, open its parameters, change one, and save*. Before R the first
	 * and the last were impossible — the TopBar's intrinsic minimum was ≈628px, so Save, Load and the
	 * add-node door were all off-screen, and the layout tab strip was squeezed to nothing beside them.
	 *
	 * Every step goes through the REAL touch surface: a long press to open the add-node menu, taps to
	 * pick and place, a tap on a connector pill to seed the next node (which is how a phone makes a
	 * cable — dragging one is a fine-pointer gesture R deliberately does not re-implement), the
	 * inspector's own rendered control for the param, and the app header's own Save button for the save.
	 * The `window.goofi` façade is used only to READ the result back.
	 *
	 * Hermeticity: the save lands in a per-run temp directory removed in afterAll, and the test hands
	 * the shared backend back unnamed and empty — `resetPatch` (a manager-side `new`) is what resets
	 * `save_path` to null, which later specs assume when they click Save expecting the browser.
	 */

	let scratch = '';
	const patchName = `r-touch-${process.pid}-${Date.now()}`;

	test.beforeAll(() => {
		scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-touch-')));
	});
	test.afterAll(() => {
		fs.rmSync(scratch, { recursive: true, force: true });
	});

	/** Hand the shared backend back empty AND unnamed even when the test fails partway — a leftover
	 * node changes where the NEXT spec's `emptySpot` may press, and a leftover NAME turns its Save
	 * into a silent overwrite. The journey saves the patch, so both are reachable failure states. */
	test.afterEach(async ({ page }) => {
		await page.evaluate(() => (window as any).goofi.commands.newPatch()).catch(() => {});
	});

	function links(page: Page): Promise<Array<Record<string, string>>> {
		return page.evaluate(() => (window as any).goofi.query.graph().links);
	}

	/**
	 * Pick `type` from the open add-node menu and place it with a tap on the canvas. Answers the
	 * GHOST's rendered size, measured while it still exists: the finger carries the ghost by its middle,
	 * so the corner the node commits at is half of that up and left of the tap — and a placed card is
	 * NOT the same height as its ghost (it grows a viewer body), so its own box cannot stand in.
	 */
	async function pickAndPlace(
		page: Page,
		type: string,
		at: { x: number; y: number }
	): Promise<{ width: number; height: number }> {
		await paletteItem(page, type).tap();
		const ghost = page.getByTestId('placement-ghost');
		await expect(ghost, `the ${type} ghost is following the finger`).toBeVisible();
		const size = (await ghost.boundingBox())!;
		await page.touchscreen.tap(at.x, at.y);
		await expect(ghost, 'the tap committed the placement').toHaveCount(0);
		return size;
	}

	test('412px portrait: add a node, connect it, open its parameters, change one, and save', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.

		// --- 1. ADD, through the coarse-pointer door -------------------------------------------
		const spot = await openAddMenuByPress(page);
		const ghost = await pickAndPlace(page, 'Oscillator', spot);
		await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);

		const osc = (await nodes(page))[0];
		const oscCard = page.locator(`.svelte-flow__node[data-id="${osc.uid}"]`);
		await expect(oscCard, 'the new node is on screen').toBeVisible();
		// …and it landed where the finger did. A touch device never reports a mouse position, so a
		// placement anchored on the last `mousemove` would drop every node at the viewport origin.
		// The finger holds the ghost by its MIDDLE (`touchPlacement.ts`'s `ghostOrigin`), so the corner
		// it commits at is half a GHOST up and left of the tap — `touch-placement.spec.ts` is where that
		// anchor is measured to the px; this is the end-to-end sanity check around it.
		const card = (await oscCard.boundingBox())!;
		expect(Math.abs(card.x - (spot.x - ghost.width / 2)), 'placed at the tap, horizontally').toBeLessThan(40);
		expect(Math.abs(card.y - (spot.y - ghost.height / 2)), 'placed at the tap, vertically').toBeLessThan(40);

		// --- 2. PARAMETERS — placing a node selects it, so its inspector is already up ------------
		const inspector = page.getByTestId('auto-side-panel');
		await expect(inspector, 'the placed node is inspected').toHaveClass(/open/);
		// This panel is portrait, so the pane is the bottom sheet: it spans the host's width and stops at
		// 60% of its height (D-I6), which means the canvas it leaves is ABOVE it rather than beside it.
		// Height is the safe read mid-slide too — a Y slide moves the box without resizing it.
		const box = (await inspector.boundingBox())!;
		const host = (await page.locator('.editor-panel').first().boundingBox())!;
		expect(box.height, 'and it leaves a strip of canvas above it').toBeLessThan(host.height);
		await expect(inspector.getByTestId('node-name')).toHaveText(osc.name);

		// --- 3. CHANGE ONE, through the rendered control -----------------------------------------
		const amp = inspector.getByTestId('param-field-amplitude').getByTestId('param-number');
		await amp.fill('0.37');
		await amp.press('Enter');
		await expect
			.poll(async () => (await nodeParams(page, osc.uid))?.oscillator?.amplitude?.value)
			.toBeCloseTo(0.37, 5);

		// --- 4. CONNECT — dismiss the inspector, then seed the next node from the output pill -----
		// The dismiss is load-bearing here, not decoration: at 412px the pane covers the canvas, so
		// without a way out the patch could not be grown past its first node (D-R9).
		await inspector.getByTestId('inspector-close').tap();
		await expect(inspector).not.toHaveClass(/open/);
		await expect(inspector).toHaveCSS('visibility', 'hidden');
		await oscCard.getByTestId('slot-output-pin').first().tap();
		await expect(page.getByTestId('add-menu-seed'), 'the menu opened seeded from that slot').toBeVisible();
		await pickAndPlace(page, 'Buffer', { x: spot.x, y: spot.y + 140 });
		await expect
			.poll(async () => (await links(page)).length, { message: 'the pick auto-wired the cable' })
			.toBe(1);
		expect((await links(page))[0].node_out).toBe(osc.uid);

		// --- 5. SAVE — from the bar, which is where Save lives at this width ----------------------
		// It used to spill, and this step used to reach it through the overflow menu. What moved is not
		// the overflow: the header carried an always-on "connected" chip whose 72px was exactly what
		// pushed Save out at 412px, and with the connection silent unless it BREAKS the bar keeps
		// Undo · Redo · Save · Load… here and gives up only the caret. Taking the door the user now has
		// is the point of the journey; the menu route is still pinned where it belongs —
		// `touch-reflow.spec.ts` asserts bar/menu parity at this very geometry, and
		// `topbar-overflow.spec.ts` opens the menu and reads its rows.
		const kept = await settledBar(page);
		expect(kept, 'at 412px the bar keeps Save itself; only the caret spills').toContain('topbar-save');
		await page.getByTestId('topbar-save').tap();

		const modal = page.getByTestId('fs-browser');
		await expect(modal, 'Save reached the file browser').toBeVisible();
		await expect(modal.getByTestId('fs-path-input')).not.toHaveValue('');
		const bar = modal.getByTestId('fs-path-input');
		await bar.fill(scratch);
		await bar.press('Enter');
		await expect(bar).toHaveValue(scratch);
		await modal.getByTestId('fs-filename').fill(patchName);
		await modal.getByTestId('fs-save').tap();
		await expect(modal, 'confirming Save closes the browser').toBeHidden();

		const patchFile = path.join(scratch, `${patchName}.gfi`);
		await expect.poll(() => fs.existsSync(patchFile), { message: 'the .gfi landed on disk' }).toBe(true);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges))
			.toBe(false);

		// --- hand the shared backend back unnamed and empty ---------------------------------------
		// One `new` does both halves; the `afterEach` below is the backstop for a failure before here.
		await resetPatch(page);
	});
});
