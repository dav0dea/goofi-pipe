import { expect, type Locator, type Page } from '@playwright/test';
import { emptySpot, touchSession, type TouchPoint } from './touch';

/**
 * Getting a placement ghost onto the canvas through the REAL coarse-pointer door.
 *
 * Two specs need it — `touch-authoring` (R's headline: add a node at 412px) and `touch-placement`
 * (the ghost's own gesture) — and a spec cannot import another spec, since loading a spec file is
 * how Playwright registers its tests. So it lives here, in one copy: the palette row selector in
 * particular is fiddly enough that two of it would drift.
 */

/** One row of the open add-node menu, by node type. */
export function paletteItem(page: Page, type: string): Locator {
	return page
		.getByTestId('add-menu-list')
		.locator('.item')
		.filter({ has: page.locator('.t-name', { hasText: new RegExp(`^${type}$`) }) })
		.first();
}

/**
 * Long-press bare canvas to open the add-node menu, and answer where the press landed.
 *
 * The long press is the ONLY route to this menu on a phone — the other three (double-click, Tab, a
 * port click) are all fine-pointer or keyboard — so a spec that reaches it any other way is not
 * testing what a phone can do.
 */
export async function openAddMenuByPress(page: Page): Promise<TouchPoint> {
	const spot = await emptySpot(page);
	const touch = await touchSession(page);
	await touch.down(spot);
	await expect(page.getByTestId('add-node-menu-anchor'), 'the long press opened the menu').toBeVisible();
	await touch.up();
	return spot;
}

/** Press, pick `type`, and leave the ghost pending on the canvas. */
export async function openGhost(page: Page, type: string): Promise<Locator> {
	await openAddMenuByPress(page);
	await paletteItem(page, type).tap();
	const ghost = page.getByTestId('placement-ghost');
	await expect(ghost, `the ${type} ghost is pending`).toBeVisible();
	return ghost;
}
