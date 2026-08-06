import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { controlsInset } from '../lib/editor';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { emptySpot, touchSession } from '../lib/touch';

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
	await expect(page.getByTestId('auto-side-panel')).toHaveCount(0);
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
	await expect(page.getByTestId('auto-side-panel')).toHaveCount(0);
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
