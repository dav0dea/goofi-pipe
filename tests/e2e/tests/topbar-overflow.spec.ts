import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp } from '../lib/app';
import { AS_ROWS, PRIORITY, inBar, menuRow, openOverflow, settledBar } from '../lib/topbar';

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
 * How many actions spill depends on the status cluster's width, and that depends on whether
 * the patch has a NAME — which this file never established. It reached three only because
 * `dirty-taxonomy.spec.ts` and `fs-browser.spec.ts` sort earlier in the single-worker `default`
 * project and left the backend in a state nothing here asked for; standalone it measured two. So
 * every test that needs a spill to EXIST brings its own precondition and hands the backend back
 * unnamed (a `loadText` — a load with no path — is what resets `save_path` to null).
 *
 * That is four tests now: two of them joined the list when the header stopped carrying a connection
 * chip. "Connected" was up at every moment of the app's life, and its 72px was quietly what pushed
 * an untitled patch's bar over the edge at 320px. Nothing about the overflow changed — the bar
 * simply has 72px more to work with, so width alone no longer guarantees the spill those two are
 * written to observe.
 *
 * The rest of the file is order-independent on purpose: declared order and bar/menu parity are
 * statements about a bar at whatever width, not about a particular one.
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

test('actions leave one at a time, lowest priority first — and the identity leaves before them', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		const left: string[] = [];
		let prev = new Set<string>(PRIORITY);
		let pathLeftAt = 0;
		let firstActionLeftAt = 0;
		// Down to 280, not 320: the identity yields ~200px before any action does, so a named
		// patch no longer forces an ACTION spill at 320 — that is the feature, not slack.
		for (let w = 1400; w >= 280; w -= 20) {
			await widthTo(page, w);
			const now = new Set(await inBar(page));
			for (const id of now) {
				expect(prev.has(id), `${id} came BACK into the bar at ${w}px`).toBe(true);
			}
			for (const id of prev) if (!now.has(id)) left.push(id);
			if (!firstActionLeftAt && left.length) firstActionLeftAt = w;
			if (!pathLeftAt && (await page.getByTestId('topbar-path').isHidden())) pathLeftAt = w;
			prev = now;
		}
		expect(left.length, 'the bar does give actions up as it narrows').toBeGreaterThanOrEqual(1);
		expect(left, 'and it gives them up in the declared priority order').toEqual(
			SPILL_ORDER.slice(0, left.length)
		);
		expect(pathLeftAt, 'the patch identity yielded on the way').toBeGreaterThan(0);
		expect(
			pathLeftAt,
			'…and it yielded BEFORE the first action did (informational outranks nothing)'
		).toBeGreaterThan(firstActionLeftAt);
	});
});

test('every spilled action is reachable in the overflow menu — and only those', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		// 280, not 320: with the identity yielding its ~200px first, a named patch keeps all five
		// actions at 320 — the mixed state this test wants (some actions in the bar, some in the
		// menu, identity already gone) starts a step lower.
		await widthTo(page, 280);
		// SETTLED, not a single read: at a named patch the plan converges over several observer
		// rounds (the cluster and the strip give way together), and a read taken mid-flight reports
		// an action as kept one frame before it is `display: none` — which then fails against the
		// menu, where it has already arrived. Caught at --repeat-each=20, ~10% of runs.
		const kept = await settledBar(page);
		expect(kept.length, 'something spilled at 280px').toBeLessThan(PRIORITY.length);
		await expect(
			page.getByTestId('topbar-path'),
			'the identity spilled long before this width'
		).toBeHidden();

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
});

/* Trap 1, at the real bar rather than at the arithmetic: moving an item out changes the bar's
   content, which re-fires the observer that moved it. A plan that read the bar's own width would
   flip-flop forever at exactly the boundary. */
test('the bar is stable across a boundary width, in both directions', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		// Find the first width at which an ACTION spills, then straddle it. Down to 260: the
		// identity yields first, so the action boundary sits lower than it used to.
		let boundary = 0;
		for (let w = 1400; w >= 260; w -= 4) {
			await widthTo(page, w);
			if ((await inBar(page)).length < PRIORITY.length) {
				boundary = w;
				break;
			}
		}
		expect(boundary, 'the bar does overflow somewhere between 260 and 1400').toBeGreaterThan(0);

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
	// The check mark and the per-row icons are decoration: the row's NAME is its label alone.
	await expect(off).toHaveAccessibleName('Multi-select mode');
	await off.click();
	await expect(trigger, 'the always-visible chrome carries the mode').toHaveClass(/multi-on/);
	await expect(trigger).toHaveAttribute('title', /multi-select mode is on/);

	// …and the row itself reads back as checked next time the menu is opened.
	await openOverflow(page);
	const row = menuRow(page, 'Multi-select mode');
	await expect(row.locator('.check svg')).toHaveAttribute('data-icon', 'check');
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
   the status cluster and the tab strip shrink TOGETHER, so two adjacent spill sets can both be
   self-consistent and the settled bar could depend on which side the width was approached from.
   The boundary test above only ever approaches from one step away; this walks in from 40px out on
   each side, at a named patch (the state where the cluster is widest and the band, if there is one,
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

/* ---- The right-bounded identity group (Phil, 2026-08-08) --------------------------------------
 * The layout tab strip is the header's first-class citizen: it owns the LEFT edge and its content
 * gets space-priority. The patch identity — perf HUD, filename, dirty dot — moves to the RIGHT and
 * joins the progressive overflow: it spills into the ⋯ menu BEFORE any action does, as disabled
 * informational rows. The connection chip stays out of the plan entirely (a warning that overflows
 * into a hidden menu is not a warning). */

test('the tab strip owns the left edge; the patch identity sits right, in the overflow group', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const first = page.locator('.topbar > :first-child');
	await expect(first, 'the tab strip is the first section of the bar').toHaveClass(/tabslot/);
	await withNamedPatch(page, async () => {
		const zone = page.locator('.topbar .action-zone');
		await expect(
			zone.locator('[data-testid="topbar-path"]'),
			'the patch name is a resident of the right-hand group'
		).toBeVisible();
		const slot = (await page.locator('.topbar .tabslot').boundingBox())!;
		const p = (await zone.locator('[data-testid="topbar-path"]').boundingBox())!;
		expect(p.x, 'name to the right of the strip').toBeGreaterThan(slot.x + slot.width - 1);
	});
});

test('the identity yields to the menu before any action does', async ({ page }) => {
	// 430px: too narrow for the (32ch-capped) name AND the actions, wide enough for the actions
	// alone — the exact band where the priority order is observable.
	await page.setViewportSize({ width: 430, height: 720 });
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		await expect(page.locator('.topbar [data-testid="topbar-path"]')).toBeHidden();
		const kept = await settledBar(page);
		// Set-compare: `inBar` reads DOM order (undo·redo·save·caret·load), PRIORITY is spill
		// order — same members, different sequences.
		expect(new Set(kept), 'every action outranks the identity').toEqual(new Set(PRIORITY));
		await openOverflow(page);
		try {
			const row = page.locator('.context-menu .item', { hasText: CROWDING_NAME.slice(0, 24) });
			await expect(row, 'the name is a row in the menu instead').toHaveCount(1);
			await expect(row, 'informational, not clickable').toBeDisabled();
		} finally {
			await page.keyboard.press('Escape');
		}
	});
});

test('layout tabs get the room the bar gives up', async ({ page }) => {
	await page.setViewportSize({ width: 800, height: 720 });
	await page.goto('/');
	await waitForApp(page);
	await withNamedPatch(page, async () => {
		// At 800px the name chip fits comfortably beside every action…
		await expect(page.getByTestId('topbar-path')).toBeVisible();
		// …then SIX more layout tabs claim the width. Six is measured, not guessed: at 800px the
		// full row (chips + actions + trigger + gaps) needs ~370px against a ~772px budget-before-
		// reserve, and each "Layout N" pill costs ~60px on the ~87px single-tab strip — so the
		// identity's spill boundary sits at seven tabs, with ~50px of margin against the rem
		// clamp. (Four and five both landed inside the noise, which is how this comment earned
		// its arithmetic.) The reserve follows the strip's CONTENT, so the identity yields into
		// the menu in the tabs' favour — and the strip fits whole instead of scrolling while
		// chips sit comfortable.
		for (let i = 0; i < 6; i++) {
			await page.getByRole('button', { name: 'New tab' }).click();
		}
		await expect(
			page.getByTestId('topbar-path'),
			'the identity handed its width to the tabs'
		).toBeHidden();
		await expect
			.poll(async () => {
				const m = await page
					.locator('[data-testid="workspace-tabs"]')
					.evaluate((el) => ({ content: el.scrollWidth, box: el.clientWidth }));
				return m.content <= m.box + 1;
			}, 'the strip fits its content — nothing is scrolled away')
			.toBe(true);
		// The actions keep their slots at this width — the room came out of the identity first,
		// which is the declared order doing its job.
		expect(new Set(await settledBar(page))).toEqual(new Set(PRIORITY));
	});
	// `withNamedPatch`'s finally re-loads the patch saved BEFORE the tabs were added, so the
	// layout collapses back to the single tab later specs assume.
});
