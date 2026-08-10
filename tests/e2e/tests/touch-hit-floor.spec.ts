import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { dismiss, spawnSh } from '../lib/harness';

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
			await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
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
 * which ANY product class rule out-specifies. Four did; one test each, because each needs its own
 * panel or its own gesture and one long test times out where four short ones do not.
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
	// A Python node whose process() needs a connected ARRAY input raises every tick; the graph store
	// mirrors each raise into the console. The cheapest real console content.
	const uid = await addNode(page, 'LempelZiv', 'python');
	try {
		await waitForNode(page, uid);
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
