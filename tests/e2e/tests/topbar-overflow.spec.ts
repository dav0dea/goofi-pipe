import { test, expect, type Page } from '@playwright/test';
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
	await expect(trigger).toHaveAttribute('aria-pressed', 'false');

	await openOverflow(page);
	await menuRow(page, 'Multi-select mode').click();
	await expect(trigger, 'the always-visible chrome carries the mode').toHaveAttribute(
		'aria-pressed',
		'true'
	);

	// …and the row itself reads back as checked next time the menu is opened.
	await openOverflow(page);
	const row = menuRow(page, 'Multi-select mode');
	await expect(row.locator('.check')).toHaveText('✓');
	await row.click();
	await expect(trigger).toHaveAttribute('aria-pressed', 'false');
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
	} finally {
		await openOverflow(page);
		await menuRow(page, 'Multi-select mode').click();
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes(u), uids);
	}
});
