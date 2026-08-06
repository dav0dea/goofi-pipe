import { expect, type Page } from '@playwright/test';
import { addNode, waitForNode, waitForNoNode } from './goofi';

/**
 * The stage both halves of the input-name proximity reveal are measured on: one source and two
 * sinks, the second far enough from the first that a reveal reaching it would be a defect.
 *
 * Shared through `lib/` rather than by importing one spec from the other, since loading a spec file
 * is how Playwright registers its tests — and the geometry is the assertion here, so a second copy
 * would be a second thing to keep true.
 */
export interface ProximityStage {
	/** The Oscillator whose output the cable is dragged from. */
	src: string;
	/** The Buffer the cable is dragged TOWARD. */
	near: string;
	/** A second Buffer, far enough away that its name must stay hidden. */
	far: string;
}

/** Flow-space rows. 160 flow px apart is ~136 screen px at the default 0.85 zoom — comfortably
 *  outside `SLOT_PROXIMITY_PX` (44 screen px) while keeping all three nodes on a 412px phone. */
export async function proximityStage(page: Page): Promise<ProximityStage> {
	const src = await addNode(page, 'Oscillator', 'inputs', [30, 40]);
	const near = await addNode(page, 'Buffer', 'signal', [300, 40]);
	const far = await addNode(page, 'Buffer', 'signal', [300, 200]);
	for (const uid of [src, near, far]) await waitForNode(page, uid);
	for (const uid of [src, near, far])
		await expect(page.locator(`.svelte-flow__node[data-id="${uid}"]`)).toBeVisible();
	return { src, near, far };
}

export async function clearStage(page: Page, stage: ProximityStage): Promise<void> {
	for (const uid of [stage.src, stage.near, stage.far]) {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
}

/** The `.conn-label` of a node's first input connector — the only rendering of that slot's name. */
export const inputLabel = (page: Page, uid: string) =>
	page.locator(`.svelte-flow__node[data-id="${uid}"] .conn.in .conn-label`).first();

const centre = async (page: Page, sel: string): Promise<{ x: number; y: number }> => {
	const box = await page.locator(sel).first().boundingBox();
	expect(box, `${sel} is on screen`).toBeTruthy();
	return { x: Math.round(box!.x + box!.width / 2), y: Math.round(box!.y + box!.height / 2) };
};

/** Where a cable drag starts: the source node's output HANDLE (SvelteFlow binds the connection to
 *  the handle itself, not to the connector pill goofi draws around it). */
export const outHandle = (page: Page, uid: string) =>
	centre(page, `.svelte-flow__node[data-id="${uid}"] .svelte-flow__handle-right`);

/** Where it is aimed: the target node's input connector. */
export const inConn = (page: Page, uid: string) =>
	centre(page, `.svelte-flow__node[data-id="${uid}"] .conn.in`);

/** A point on bare canvas to release a cable over, so no link is actually made. */
export async function bareSpot(page: Page): Promise<{ x: number; y: number }> {
	const spot = await page.locator('.svelte-flow__pane').first().evaluate((pane) => {
		const cards = [...document.querySelectorAll('.svelte-flow__node')].map((n) =>
			n.getBoundingClientRect()
		);
		const clear = (x: number, y: number): boolean =>
			cards.every((c) => x < c.left - 80 || x > c.right + 80 || y < c.top - 80 || y > c.bottom + 80);
		const r = pane.getBoundingClientRect();
		for (let fy = 0.8; fy >= 0.3; fy -= 0.1)
			for (let fx = 0.15; fx <= 0.85; fx += 0.1) {
				const x = Math.round(r.left + r.width * fx);
				const y = Math.round(r.top + r.height * fy);
				if (document.elementFromPoint(x, y) === pane && clear(x, y)) return { x, y };
			}
		return null;
	});
	expect(spot, 'the canvas has somewhere to drop a cable harmlessly').not.toBeNull();
	return spot!;
}
