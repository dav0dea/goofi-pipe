import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp } from '../lib/app';
import { AS_ROWS, PRIORITY, inBar, menuRow, openOverflow } from '../lib/topbar';

/**
 * The app header's progressive overflow (D-R6) and the canvas commands it carries (D-R4).
 *
 * This is a DESKTOP spec on purpose. The collapse keys on available WIDTH, not on device class —
 * a 900px desktop window has exactly the phone's problem — and the menu is resident chrome at
 * every width, because the commands inside it (delete, group, select-all, copy/paste/duplicate,
 * multi-select) have no bar slot to lose and are otherwise reachable only by a keyboard chord.
 *
 * The three traps D-R6 names are pinned as arithmetic in `editor/overflowFit.test.ts`; what this
 * file proves is that the real bar is wired to that arithmetic — that items really leave in the
 * declared order, that the result is stable when the width crosses a boundary and comes back, and
 * that nothing the header used to do changed.
 */

/** The bar gives its actions up in reverse priority order. (`PRIORITY`, `AS_ROWS` and the
 * handles for reading the bar live in `lib/topbar.ts` — `touch-reflow.spec.ts` asks the same
 * question of the same list at three real device geometries.) */
const SPILL_ORDER = [...PRIORITY].reverse();

/**
 * How many actions spill depends on the brand's width, and the brand's width depends on whether
 * the patch has a NAME — which this file never established. It reached three only because
 * `dirty-taxonomy.spec.ts` and `fs-browser.spec.ts` sort earlier in the single-worker `default`
 * project and left the backend in a state nothing here asked for; standalone it measured two. So
 * the one test that counts spills brings its own precondition and hands the backend back unnamed
 * (a `loadText` — a load with no path — is what resets `save_path` to null).
 *
 * The rest of the file is order-independent on purpose: declared order, boundary stability and
 * bar/menu parity are all statements about a bar at whatever width, not about a particular one.
 */
const CROWDING_NAME = 'a-patch-with-a-deliberately-long-name-that-crowds-the-header';
let scratch = '';

test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-overflow-')));
});
test.afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }));

async function withNamedPatch(page: Page, body: () => Promise<void>): Promise<void> {
	const file = path.join(scratch, `${CROWDING_NAME}.gfi`);
	await page.evaluate((p) => (window as any).goofi.commands.save(p), file);
	await expect(page.locator('.topbar .path'), 'the header is showing the patch name').toHaveText(
		`${CROWDING_NAME}.gfi`
	);
	try {
		await body();
	} finally {
		await page.evaluate(
			(y) => (window as any).goofi.commands.loadText(y),
			fs.readFileSync(file, 'utf8')
		);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.toBe(null);
	}
}

/** Resize and let the ResizeObserver settle (it runs after layout, before the next paint). */
async function widthTo(page: Page, width: number): Promise<void> {
	await page.setViewportSize({ width, height: 720 });
	await page.evaluate(
		() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
	);
}

test('the overflow trigger is resident chrome, outside the pinned action list', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const trigger = page.getByTestId('topbar-overflow');
	await expect(trigger).toBeVisible();
	// `topbar.spec.ts` pins `.topbar .actions button` as an exact ordered list of the five
	// app-global actions. The trigger is not one of them, so it must not live in that box.
	await expect(page.locator('.topbar .actions [data-testid="topbar-overflow"]')).toHaveCount(0);
});

test('a wide window keeps every action in the bar', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);
	expect((await inBar(page)).sort()).toEqual([...PRIORITY].sort());
});

test('actions leave one at a time, lowest priority first', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		const left: string[] = [];
		let prev = new Set<string>(PRIORITY);
		for (let w = 1400; w >= 320; w -= 20) {
			await widthTo(page, w);
			const now = new Set(await inBar(page));
			for (const id of now) {
				expect(prev.has(id), `${id} came BACK into the bar at ${w}px`).toBe(true);
			}
			for (const id of prev) if (!now.has(id)) left.push(id);
			prev = now;
		}
		// One at a time, in the declared order — a prefix of it, since how many fit at 320px depends
		// on the root size the responsive clamp lands on, which is not this test's question.
		expect(left.length, 'the bar does give actions up as it narrows').toBeGreaterThanOrEqual(3);
		expect(left, 'and it gives them up in the declared priority order').toEqual(
			SPILL_ORDER.slice(0, left.length)
		);
	});
});

test('every spilled action is reachable in the overflow menu — and only those', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 320);
	const kept = await inBar(page);
	expect(kept.length, 'something spilled at 320px').toBeLessThan(PRIORITY.length);

	await openOverflow(page);
	for (const id of PRIORITY) {
		const spilledHere = !kept.includes(id);
		for (const label of AS_ROWS[id]) {
			// Present exactly when the bar gave it up: a row that duplicates a visible button is
			// two doors onto one action, which is how the two representations drift apart.
			await expect(menuRow(page, label), `${label} (${id} spilled: ${spilledHere})`).toHaveCount(
				spilledHere ? 1 : 0
			);
		}
	}
});

/* Trap 1, at the real bar rather than at the arithmetic: moving an item out changes the bar's
   content, which re-fires the observer that moved it. A plan that read the bar's own width would
   flip-flop forever at exactly the boundary. */
test('the bar is stable across a boundary width, in both directions', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	// Find the first width at which something spills, then straddle it.
	let boundary = 0;
	for (let w = 1400; w >= 320; w -= 4) {
		await widthTo(page, w);
		if ((await inBar(page)).length < PRIORITY.length) {
			boundary = w;
			break;
		}
	}
	expect(boundary, 'the bar does overflow somewhere between 320 and 1400').toBeGreaterThan(0);

	await widthTo(page, boundary);
	const at = (await inBar(page)).join();
	await widthTo(page, boundary + 24);
	const above = (await inBar(page)).join();
	for (let i = 0; i < 4; i++) {
		await widthTo(page, boundary);
		expect(await inBar(page), 'crossing back lands on the same answer').toEqual(at.split(','));
		await widthTo(page, boundary + 24);
		expect(await inBar(page)).toEqual(above.split(','));
	}

	// …and having settled, it stays settled with nobody touching it.
	await widthTo(page, boundary);
	await page.waitForTimeout(500);
	expect(await inBar(page)).toEqual(at.split(','));
});

test('the layout tab strip keeps a floor to spill against', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 412);
	// o1: the actions used to take every pixel and squeeze `.tabslot` to zero, so every layout tab
	// AND the ＋ that makes one were unreachable. Giving the actions a floor to spill below is
	// what fixes it.
	const slot = await page.locator('.topbar .tabslot').boundingBox();
	expect(slot!.width, 'the tab strip is not squeezed to nothing').toBeGreaterThan(40);
	await expect(page.getByRole('button', { name: 'New tab' })).toBeVisible();
});

/**
 * R's §5.2 for the chord half: no interaction may exist solely behind a keyboard chord. This list
 * is the editor's chord inventory (`NodeEditorPanel.onKeydown`) minus the three that already have
 * a pointer door of their own — Tab (long-press the canvas), F (the panel's own Fit control) and
 * Escape (a tap on the canvas, or the inspector's ✕). Everything else is here, at every width,
 * because the menu is resident chrome rather than a collapse target.
 */
const CHORD_ROWS = [
	'Select all', // ⌘A
	'Delete selection', // Delete
	'Group into sub-patch', // ⌘G
	'Copy', // ⌘C
	'Paste', // ⌘V
	'Duplicate', // ⌘D
	'Multi-select mode' // no chord at all — the touch door for shift-click
];

test('the canvas commands live in the menu at every width', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	for (const w of [1400, 412]) {
		await widthTo(page, w);
		await openOverflow(page);
		for (const name of CHORD_ROWS) {
			await expect(menuRow(page, name), `${name} at ${w}px`).toBeVisible();
		}
		await page.keyboard.press('Escape');
	}
});

test('Select all from the menu selects the editor’s nodes', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await page.evaluate(() =>
		(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
	);
	await page.waitForFunction(
		(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
		uid
	);
	try {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await openOverflow(page);
		await menuRow(page, 'Select all').click();
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
			.toBeGreaterThan(0);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
	}
});

test('multi-select mode is a mode: it stays on, and the header says so', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const trigger = page.getByTestId('topbar-overflow');
	// The trigger says the ONE thing it owns and changes on activation: whether the menu is open.
	// The mode is not its state — it neither owns it nor toggles it — so it carries the mode only
	// as the visible tell (the accent + the title), and the row that toggles it says `aria-checked`.
	await expect(trigger).toHaveAttribute('aria-expanded', 'false');

	await openOverflow(page);
	await expect(trigger).toHaveAttribute('aria-expanded', 'true');
	const off = menuRow(page, 'Multi-select mode');
	await expect(off, 'a checkable row is a checkbox, not a plain item').toHaveAttribute(
		'aria-checked',
		'false'
	);
	// The ✓ and the ▦ / ✕ / ▣ glyphs are decoration: the row's NAME is its label alone.
	await expect(off).toHaveAccessibleName('Multi-select mode');
	await off.click();
	await expect(trigger, 'the always-visible chrome carries the mode').toHaveClass(/multi-on/);
	await expect(trigger).toHaveAttribute('title', /multi-select mode is on/);

	// …and the row itself reads back as checked next time the menu is opened.
	await openOverflow(page);
	const row = menuRow(page, 'Multi-select mode');
	await expect(row.locator('.check')).toHaveText('✓');
	await expect(row).toHaveAttribute('aria-checked', 'true');
	await expect(row).toHaveAccessibleName('Multi-select mode');
	await row.click();
	await expect(trigger).not.toHaveClass(/multi-on/);
});

/* One label column per menu. `.check` used to render on every row while `.ic` rendered only where
   there was an icon, so the one menu that mixes them — this one — laid its labels out in two
   columns 18px apart, on the phone and on the desktop alike. */
test('every row in a menu starts its label in the same column', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await openOverflow(page);
	const columns = await page
		.locator('.context-menu .item .label')
		.evaluateAll((els) => [...new Set(els.map((e) => Math.round(e.getBoundingClientRect().x)))]);
	expect(columns, 'the labels share one x').toHaveLength(1);
});

test('with multi-select on, a plain click adds instead of replacing', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uids: string[] = [];
	// Apart, and to the LEFT: two nodes at one point would cover each other's click target, and the
	// inspector that opens on the first selection takes the right of the canvas.
	for (const [type, cat, x] of [
		['Oscillator', 'inputs', 0],
		['Buffer', 'signal', 260]
	] as const) {
		const uid = await page.evaluate(
			([t, c, px]) => (window as any).goofi.commands.addNode(t, c, [px as number, 0]),
			[type, cat, x] as const
		);
		await page.waitForFunction(
			(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
			uid
		);
		uids.push(uid);
	}
	const selected = () =>
		page.evaluate(() => (window as any).goofi.query.selection().nodes as string[]);
	try {
		// Baseline: without the mode a second plain click REPLACES — the behaviour a phone is stuck
		// with, since it has no shift, ctrl or meta.
		for (const u of uids) await page.locator(`.svelte-flow__node[data-id="${u}"]`).click();
		await expect.poll(selected).toEqual([uids[1]]);

		await openOverflow(page);
		await menuRow(page, 'Multi-select mode').click();
		// From empty: with a node already selected the first click would TOGGLE it back off, which
		// is the same additive semantics shift-click has and not what this case is about.
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		for (const u of uids) await page.locator(`.svelte-flow__node[data-id="${u}"]`).click();
		await expect.poll(async () => (await selected()).slice().sort()).toEqual([...uids].sort());

		// …and a plain click on empty canvas must not wipe it. `clickPane` took the same fold as
		// clickNode/clickEdge: on a phone `shiftKey` is always false, so the biggest target on
		// screen was undoing the very selection the mode exists to build.
		// Found rather than guessed: the pane's corners are taken (the zoom cluster sits in one) and
		// the nodes float wherever the fit put them, so scan for a point that really is bare canvas.
		const empty = await page.evaluate(() => {
			const r = document.querySelector('.svelte-flow__pane')!.getBoundingClientRect();
			for (let y = r.bottom - 20; y > r.top; y -= 20)
				for (let x = r.left + 20; x < r.right; x += 20)
					if (document.elementFromPoint(x, y)?.classList.contains('svelte-flow__pane'))
						return { x, y };
			return null;
		});
		expect(empty, 'the canvas has some bare pane to click').not.toBeNull();
		await page.mouse.click(empty!.x, empty!.y);
		await expect
			.poll(async () => (await selected()).slice().sort(), {
				message: 'a tap on empty canvas leaves the multi-selection alone'
			})
			.toEqual([...uids].sort());
		// …and the CANVAS agrees. SvelteFlow unselects on every pane click, after the callback and
		// whatever the store decided, so the store keeping the selection is only half the fix: the
		// other half is that nothing gated on the rendered flags (Delete, Group, Copy, this row)
		// goes dead while the store still holds the operands.
		await expect(
			page.locator('.svelte-flow__node.selected'),
			'the nodes are still painted selected'
		).toHaveCount(2);

		// So the way out has to be somewhere a pointer can reach — Escape is a keyboard's door only.
		await openOverflow(page);
		await menuRow(page, 'Clear selection').click();
		await expect.poll(selected, { message: 'the menu row is the mode’s deselect' }).toEqual([]);
	} finally {
		await openOverflow(page);
		await menuRow(page, 'Multi-select mode').click();
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes(u), uids);
	}
});

/* The hysteresis probe R's audit asked for: where the header's flex line is already overflowing,
   the brand and the tab strip shrink TOGETHER, so two adjacent spill sets can both be
   self-consistent and the settled bar could depend on which side the width was approached from.
   The boundary test above only ever approaches from one step away; this walks in from 40px out on
   each side, at a named patch (the state where the brand is widest and the band, if there is one,
   is widest with it). */
test('one width has one answer, whichever side it is approached from', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		for (let w = 1000; w >= 360; w -= 20) {
			await widthTo(page, w - 40);
			await widthTo(page, w);
			const fromBelow = (await inBar(page)).join();
			await widthTo(page, w + 40);
			await widthTo(page, w);
			expect((await inBar(page)).join(), `at ${w}px, approached from either side`).toBe(fromBelow);
		}
	});
});

/**
 * Escape belongs to the surface that is dismissing, and to nothing behind it.
 *
 * `ContextMenu` claims Escape on a window keydown listener and does not consume it, while
 * `NodeEditorPanel` registers its own window listener in `onMount` — long before any menu can
 * mount — so both bubble-phase handlers run and the editor's goes FIRST. Its guards do not
 * exclude the case: `TopBar` renders outside the panel tree so `activePanelId` never changes, and
 * the keydown's target is the overflow `<button>`, which is neither in the tag allowlist nor
 * inside a `dialog[open]`. So backing out of the menu also ran the editor's Escape ladder — the
 * same defect the file browser's `dialog[open]` guard fixed, on the surface that is the phone's
 * ONLY door to the canvas commands.
 *
 * Both rungs of that ladder are reachable from this exact menu: the rows that need a selection are
 * `disabled` without one, so a user who came for `Delete selection` demonstrably HAS one; and with
 * nothing selected the next rung pops a sub-patch level.
 */
test('Escape out of the overflow menu leaves the selection alone', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await page.evaluate(() =>
		(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
	);
	await page.waitForFunction(
		(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
		uid
	);
	try {
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		await openOverflow(page);
		// The row a user with a selection came for is live, which is what makes this the ordinary path.
		await expect(menuRow(page, 'Delete selection')).toBeEnabled();

		await page.keyboard.press('Escape');
		await expect(page.locator('.context-menu')).toHaveCount(0);
		expect(
			await page.evaluate(() => (window as any).goofi.query.selection().nodes),
			'dismissing the menu is not also the canvas’s deselect'
		).toEqual([uid]);
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
	}
});

test('Escape out of the overflow menu does not pop a sub-patch level', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await page.evaluate(() =>
		(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
	);
	await page.waitForFunction(
		(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
		uid
	);
	const inst: string = await page.evaluate(
		(u) => (window as any).goofi.commands.groupNodes([u], [140, 140]),
		uid
	);
	const crumbs = page.getByTestId('subpatch-breadcrumb');
	try {
		await page.locator(`.svelte-flow__node[data-id="${inst}"]`).dblclick();
		await expect(crumbs, 'inside the sub-patch').toBeVisible();
		// The second rung of the ladder needs an EMPTY selection — entering leaves the instance
		// selected, so clear it first.
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
			.toBe(0);

		await openOverflow(page);
		await page.keyboard.press('Escape');
		await expect(page.locator('.context-menu')).toHaveCount(0);
		await expect(crumbs, 'and the editor is still where it was').toBeVisible();
	} finally {
		if (await crumbs.isVisible())
			await crumbs.getByRole('button', { name: 'Patch', exact: true }).click();
		await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
	}
});
