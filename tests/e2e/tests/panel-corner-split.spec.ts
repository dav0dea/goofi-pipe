import { test, expect, type Page } from '@playwright/test';
import { closeSplit, waitForApp } from '../lib/app';

/**
 * The corner grip, under a mouse — the half `touch-panel-split.spec.ts` takes away.
 *
 * Blender-style: drag a panel corner inward and the panel splits along the dominant drag axis.
 * `gesture-cancel.spec.ts` covers what a CANCELLED corner drag must not leave behind, but nothing
 * covered the gesture actually working, so "hide the grips on touch" had no fine-pointer guard to
 * fail if it took the desktop gesture with it. This is that guard: the grip rests invisible, comes
 * up on hover, hit-tests, and a completed drag really restructures the workspace.
 */

/** The point 3px inside the panel body's top-right corner — inside the grip's clipped triangle. */
async function topRight(page: Page): Promise<{ x: number; y: number }> {
	const body = (await page.locator('.panel-body').first().boundingBox())!;
	return { x: Math.round(body.x + body.width - 3), y: Math.round(body.y + 3) };
}

test('a panel corner hit-tests as a grip and comes up on hover', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const at = await topRight(page);
	const hit = await page.evaluate(
		(p) => (document.elementFromPoint(p.x, p.y) as HTMLElement | null)?.className ?? '',
		at
	);
	expect(hit, 'the fine-pointer grip is the topmost box in the corner').toContain('corner');

	// It rests at opacity 0 and the body's hover brings the set of them up. Retrying, because the
	// reveal is a --dur-slow transition rather than a class flip.
	const grip = page.locator('.panel-body .corner.tr').first();
	await expect(grip).toHaveCSS('opacity', '0');
	await page.locator('.panel-body').first().hover();
	await expect(grip).not.toHaveCSS('opacity', '0');
});

test('dragging a corner inward splits the panel', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panels = page.locator('.panel');
	const at = await topRight(page);

	await page.mouse.move(at.x, at.y);
	await page.mouse.down();
	// Leftward, dominantly horizontal: a row split whose new panel takes the right-hand share.
	await page.mouse.move(at.x - 120, at.y + 8);
	await page.mouse.move(at.x - 240, at.y + 10);
	await expect(page.locator('.drag-ghost'), 'the split is previewed while dragging').toHaveCount(1);

	try {
		await page.mouse.up();
		await expect(panels, 'the release commits the split').toHaveCount(2);
		await expect(page.locator('.drag-ghost'), 'and the preview is dropped').toHaveCount(0);
	} finally {
		await closeSplit(page);
	}
});
