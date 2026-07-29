import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

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

test('a double-click enters a sub-patch parked under the inspector without dissolving it', async ({
	page
}) => {
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

		// A normal double-click: two clicks at one point, 150ms apart — long enough for the pane to
		// have finished its 120ms slide over the node.
		await page.mouse.click(target.x, target.y);
		await page.waitForTimeout(150);
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
		const crumb = page.getByTestId('subpatch-breadcrumb');
		if (await crumb.isVisible())
			await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
		if (Object.keys(await instances(page)).includes(inst))
			await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
		await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [osc, buf]);
		await waitForNoNode(page, osc).catch(() => {});
		await waitForNoNode(page, buf).catch(() => {});
	}
});
