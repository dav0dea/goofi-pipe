import { test, expect, type Locator, type Page } from '@playwright/test';
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
 *
 * And the interlock at the bottom is the point of taking the grips away at all: the panel header
 * carries Split Right and Split Down as real controls now (progressive overflow, `overflowFit.ts`),
 * so a finger that lost the corner gesture did not lose the operation. That test asserts against
 * the WORKSPACE TREE (`goofi.query.panels()`), not the DOM, because gaining a `.panel` element is
 * not the same claim as gaining a panel.
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

/* ---------------------------------------------------------------------------------------------
   The interlock: with the corner gesture gone, the header is what a finger splits with.
   -------------------------------------------------------------------------------------------- */

const hdr = (page: Page, n = 0): Locator => page.getByTestId('panel-header').nth(n);

/** How many panels the WORKSPACE TREE holds — the claim, as opposed to how many `.panel` elements
 *  happen to be mounted. */
const treePanels = (page: Page): Promise<number> =>
	page.evaluate(() => (window as any).goofi.query.panels().length as number);

/** Tap Split Down wherever the header is currently keeping it: inline while it fits, and behind the
 *  ⋯ once the panel is too narrow for it. Both are the same command (D-R2), and a phone will meet
 *  whichever the width decides — so the door the test uses is the one the header is offering. */
async function splitDownFromHeader(page: Page): Promise<void> {
	const inline = hdr(page).getByTestId('panel-split-column');
	if (await inline.isVisible()) {
		await inline.tap();
		return;
	}
	await hdr(page).getByTestId('panel-overflow').tap();
	await expect(page.locator('.context-menu').first()).toBeVisible();
	await page
		.locator('.context-menu .item')
		.filter({ has: page.locator('.label', { hasText: /^Split Down$/ }) })
		.tap();
}

test('a finger can still split a panel — the header carries what the corner gave up', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	expect(await treePanels(page), 'the workspace starts as one panel').toBe(1);

	try {
		await splitDownFromHeader(page);
		await expect
			.poll(() => treePanels(page), { message: 'the workspace tree really gained a panel' })
			.toBe(2);
		await expect(page.locator('.panel')).toHaveCount(2);
	} finally {
		await hdr(page, 1).getByRole('button', { name: 'Close panel' }).tap();
		await expect(page.locator('.panel'), 'the workspace is handed back').toHaveCount(1);
	}
});
