import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { touchSession, type TouchPoint } from '../lib/touch';
import {
	VIEWER_HOVER_SURFACES,
	hoverSettled,
	surfaceStyles,
	unhover
} from '../lib/viewerChrome';

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

		await tap(`${sv} select.kind`);
		expect(
			await page.evaluate(() => document.activeElement?.className ?? ''),
			'the kind picker takes the tap'
		).toContain('kind');

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
