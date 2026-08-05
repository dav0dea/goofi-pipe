import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, nodes, waitForNode, waitForNoNode } from '../lib/goofi';
import { paletteItem } from '../lib/placement';
import { emptySpot } from '../lib/touch';

/**
 * Entering a sub-patch by double-clicking its group node — the DESKTOP half (the coarse door is
 * `touch-editor.spec.ts`).
 *
 * `enterInstance` has exactly one caller: a hand-rolled double-click recogniser in
 * `NodeEditorPanel.svelte`, because neither SvelteFlow's `onnodeclick` (it suppresses the second
 * click) nor the native `dblclick` (the first click rebuilds `flowNodes` and detaches the element)
 * survives the first click. What that recogniser did NOT do was re-resolve the second click's
 * target: it matched on COORDINATES alone. So the first click selects the instance, the inspector
 * slides in over 120ms and covers all but `--hit` of the editor — including the very node the
 * second click has to hit — and the second click both entered the instance AND actuated whatever
 * inspector control had arrived under the pointer. On a sub-patch that control is
 * `subpatch-expand-inspector`, which DISSOLVES it.
 *
 * Not a touch defect: this reproduces at a normal ~150ms desktop double-click. The suite hid it
 * only because Playwright's `.dblclick()` fires both clicks ~0ms apart, before the pane has begun
 * to slide.
 */

/** The editor viewport's flow→screen scale (its CSS transform's `a` component). */
function flowScale(page: Page): Promise<number> {
	return page
		.locator('.svelte-flow__viewport')
		.first()
		.evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).a);
}

const nodeBox = (page: Page, uid: string) =>
	page.locator(`.svelte-flow__node[data-id="${uid}"]`).boundingBox();

/** The flow-node id actually on top at a viewport point, or null. */
const nodeAt = (page: Page, at: { x: number; y: number }): Promise<string | null> =>
	page.evaluate(
		(p) =>
			(document.elementFromPoint(p.x, p.y) as HTMLElement | null)
				?.closest('.svelte-flow__node')
				?.getAttribute('data-id') ?? null,
		at
	);

/** The inspector pane's current horizontal slide offset in px (0 = fully in). */
const paneOffset = (page: Page): Promise<number> =>
	page
		.getByTestId('auto-side-panel')
		.evaluate((el) => Math.round(new DOMMatrixReadOnly(getComputedStyle(el).transform).e));

/** Reposition `uid` (currently at `from` in flow space) so its rendered CENTRE lands on `target`. */
async function centreNodeOn(
	page: Page,
	uid: string,
	from: [number, number],
	target: { x: number; y: number }
): Promise<void> {
	const box = (await nodeBox(page, uid))!;
	const s = await flowScale(page);
	const next: [number, number] = [
		Math.round(from[0] + (target.x - (box.x + box.width / 2)) / s),
		Math.round(from[1] + (target.y - (box.y + box.height / 2)) / s)
	];
	await page.evaluate(
		([u, p]) => (window as any).goofi.commands.setNodePos(u, p),
		[uid, next] as const
	);
	await expect
		.poll(async () => {
			const b = await nodeBox(page, uid);
			return b ? Math.round(Math.abs(b.x + b.width / 2 - target.x)) : 999;
		}, { message: 'the group node moved under the inspector' })
		.toBeLessThanOrEqual(2);
}

/** A sub-patch with one member, plus a link across the cut so it owns a real interface. */
async function makeSubPatch(page: Page): Promise<{ osc: string; buf: string; inst: string }> {
	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);
	const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
	await waitForNode(page, buf);
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
	await expect(page.getByTestId('subpatch-node')).toBeVisible();
	return { osc, buf, inst };
}

const instances = (page: Page): Promise<Record<string, unknown>> =>
	page.evaluate(() => (window as any).goofi.query.instances());

/** The direct member uids of a sub-patch, as the editor's forest reports them. */
async function scopeMembers(page: Page, inst: string): Promise<string[]> {
	const rec = (await instances(page))[inst] as { members?: Record<string, unknown> } | undefined;
	return Object.keys(rec?.members ?? {});
}

/** Leave the sub-patch this editor is inside, if it is inside one. */
async function exitSubPatch(page: Page): Promise<void> {
	const crumb = page.getByTestId('subpatch-breadcrumb');
	if (await crumb.isVisible()) await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
}

test('a double-click enters a sub-patch parked under the inspector without dissolving it', async ({
	page
}) => {
	// The pane's 120ms slide is a hazard in BOTH directions here — it is still covering the node
	// while it parks, and still off it while it arrives — and neither is what this test is about.
	// The app's own reduced-motion rule collapses every transition to 0.01ms, so the pane's position
	// is a function of the selection alone and the click interval below is free to be a realistic
	// double-click rather than a race against a transition.
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const { osc, buf, inst } = await makeSubPatch(page);
	try {
		// Learn the inspector's footprint from the pane it opens for THIS node, then park the group
		// node exactly under its one structural action — the button a stray second click actuates.
		await page.evaluate((i) => (window as any).goofi.commands.select([i]), inst);
		const expand = page.getByTestId('subpatch-expand-inspector');
		await expect(expand).toBeVisible();
		// Measured only once the 120ms slide has SETTLED — mid-transition the pane's translateX still
		// puts the button hundreds of px off the right edge of the window.
		await expect.poll(() => paneOffset(page)).toBe(0);
		const eb = (await expand.boundingBox())!;
		const target = { x: Math.round(eb.x + eb.width / 2), y: Math.round(eb.y + eb.height / 2) };
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await expect(page.getByTestId('auto-side-panel')).not.toHaveClass(/open/);
		await centreNodeOn(page, inst, [200, 200], target);
		// Stated as a precondition, not assumed: the FIRST click has to reach the node, so the node
		// has to be what is actually on top at that point.
		await expect
			.poll(() => nodeAt(page, target), { message: 'the group node is the topmost thing there' })
			.toBe(inst);

		// A normal double-click: two clicks at one point, ~120ms apart.
		await page.mouse.click(target.x, target.y);
		await page.waitForTimeout(120);
		await page.mouse.click(target.x, target.y);

		// Settled, not sampled. Today the gesture enters the instance AND actuates Expand under the
		// pointer, and the dissolve is an RPC round-trip — so the breadcrumb really does appear for a
		// frame or two before the climb-out effect pops it. Asserting the END state is what tells the
		// two outcomes apart.
		await page.waitForTimeout(500);
		expect(
			Object.keys(await instances(page)),
			'the sub-patch survived the gesture (the inspector control under the pointer did not fire)'
		).toContain(inst);
		await expect(
			page.getByTestId('subpatch-breadcrumb'),
			'and the double-click entered it'
		).toBeVisible();
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await exitSubPatch(page);
		if (Object.keys(await instances(page)).includes(inst))
			await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
		await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [osc, buf]);
		await waitForNoNode(page, osc).catch(() => {});
		await waitForNoNode(page, buf).catch(() => {});
	}
});

/**
 * Wait until the editor camera has stopped moving.
 *
 * `enterInstance` schedules a `fitView` 60ms after it descends, and `fitView` is implemented by
 * CLICKING SvelteFlow's own Controls button — a synthetic click inside `.canvas-wrap`, which is
 * exactly what `PlacementPreview` listens for to commit. Opening the add menu before that fit has
 * landed lets it commit the ghost early, at the wrong point, and the test's own placement click then
 * falls through to the pane (clearing the selection). Settled = the viewport transform holds still,
 * sampled only after the deferred fit is due.
 */
async function cameraSettled(page: Page): Promise<void> {
	await page.waitForTimeout(150); // outlast the deferred fit; the poll below covers its animation
	let prev = '';
	await expect
		.poll(
			async () => {
				const now = await page
					.locator('.svelte-flow__viewport')
					.first()
					.evaluate((el) => getComputedStyle(el).transform);
				const held = now === prev;
				prev = now;
				return held;
			},
			{ message: 'the editor camera stopped moving' }
		)
		.toBe(true);
}

/**
 * The other half of entering: what you ADD once you are inside.
 *
 * `add_node` carried the entered scope from the panel all the way to the bridge and the bridge threw
 * it away, so the node was minted at ROOT. The canvas renders only `childrenOfScope(entered)`, so it
 * was placed into a scope that does not draw it — while the panel still auto-selected it, leaving the
 * inspector editing a node nowhere on screen. Driven through the real door (the add menu + a click to
 * place), because the auto-selection is the panel's, not the façade's.
 */
test('a node added inside an entered sub-patch lands in that scope, on screen and selected', async ({
	page
}) => {
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const { osc, buf, inst } = await makeSubPatch(page);
	let added: string | undefined;
	try {
		// Playwright's `.dblclick()` fires both clicks ~0ms apart — before the inspector can slide over
		// the node — so entering is not itself the race the spec above isolates.
		await page.getByTestId('subpatch-node').dblclick();
		await expect(page.getByTestId('subpatch-breadcrumb'), 'the sub-patch is entered').toBeVisible();
		await cameraSettled(page);

		// Add through the REAL door — the add-node menu, then a click to place the ghost — because the
		// auto-selection under test is the PANEL's, not the façade's. The type is picked from the
		// search field rather than by clicking its row: the unfiltered menu is a scrolling, grouped
		// list, so a row click is a hit-test against a box that can still be moving, and it
		// intermittently landed off the row (menu dismissed, nothing picked). Enter takes the
		// top-ranked match, asserted below, and needs no geometry at all.
		const before = new Set((await nodes(page)).map((n) => n.uid));
		await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
		const search = page.getByTestId('add-menu-search');
		await expect(search, 'the menu is open and holds the keyboard').toBeFocused();
		await search.fill('Buffer');
		await expect(
			paletteItem(page, 'Buffer'),
			'the exact name match leads the ranked list, so Enter picks it'
		).toHaveClass(/\bhl\b/);
		await search.press('Enter');

		const ghost = page.getByTestId('placement-ghost');
		await expect(ghost, 'the Buffer ghost is pending').toBeVisible();
		const spot = await emptySpot(page);
		await page.mouse.click(spot.x, spot.y);
		await expect(ghost, 'the click placed it').toHaveCount(0);

		await expect
			.poll(async () => (await nodes(page)).filter((n) => !before.has(n.uid)).length, {
				message: 'one new node landed'
			})
			.toBe(1);
		const fresh = (await nodes(page)).find((n) => !before.has(n.uid))!;
		expect(fresh.type, 'and it is the type that was picked').toBe('Buffer');
		added = fresh.uid;

		// The defect as the canvas states it: this editor draws only the entered scope's children, so
		// a node that went to ROOT is simply not here.
		await expect(
			page.locator(`.svelte-flow__node[data-id="${added}"]`),
			'the new node renders inside the entered sub-patch'
		).toBeVisible();
		// And the forest agrees it is a DIRECT member — not a root node this canvas happens to draw.
		expect(await scopeMembers(page, inst), 'a direct member of the entered scope').toContain(added);
		// The panel auto-selects what it just placed, so the inspector is showing THIS node. Polled,
		// not sampled: the panel selects only once its `add_node` RPC has resolved, and the doc echo
		// that renders the node can land first.
		await expect
			.poll(
				async () => (await page.evaluate(() => (window as any).goofi.query.selection())).nodes,
				{ message: 'the placed node is the one selected' }
			)
			.toEqual([added]);
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await exitSubPatch(page);
		if (Object.keys(await instances(page)).includes(inst))
			await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
		const leftovers = [osc, buf, ...(added ? [added] : [])];
		await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), leftovers);
		for (const uid of leftovers) await waitForNoNode(page, uid).catch(() => {});
	}
});
