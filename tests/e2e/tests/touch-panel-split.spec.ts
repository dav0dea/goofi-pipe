import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { touchSession } from '../lib/touch';

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
