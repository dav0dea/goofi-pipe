import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * The device seam's ONE surviving output (D-R8): `--kb-inset`, how far the soft keyboard overlaps
 * the layout viewport. `data-pointer` / `data-size` / `data-short` were deleted with `classify()` —
 * they were write-only, and `@media` already answers the question they encoded. `--kb-inset` is
 * kept because nothing in CSS can answer it: the overlap is observable only through
 * `visualViewport`.
 *
 * So this spec asserts BEHAVIOUR, not the presence of an attribute: the value tracks the visual
 * viewport, and the two anchored-overlay clamps place against the visual viewport rather than the
 * layout one — a menu opened with the keyboard up must not land underneath it.
 */

const MARGIN = 6; // clampToViewport's viewport-edge margin

/** Fake a soft keyboard by shrinking `visualViewport.height` (an own property shadows the
 *  prototype getter) and firing the resize the real keyboard would fire. `px = 0` restores it. */
async function setKeyboardInset(page: Page, px: number): Promise<void> {
	await page.evaluate((n) => {
		const vv = window.visualViewport as VisualViewport & { height?: number };
		if (n > 0) {
			Object.defineProperty(vv, 'height', {
				configurable: true,
				get: () => window.innerHeight - n
			});
		} else {
			delete vv.height;
		}
		vv.dispatchEvent(new Event('resize'));
	}, px);
}

const kbInset = (page: Page): Promise<string> =>
	page.evaluate(() =>
		getComputedStyle(document.documentElement).getPropertyValue('--kb-inset').trim()
	);

test('--kb-inset tracks the visual viewport, so the soft keyboard is measurable', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	expect(await kbInset(page), 'no keyboard ⇒ no inset').toBe('0px');

	await setKeyboardInset(page, 300);
	await expect.poll(() => kbInset(page), { message: 'the inset follows visualViewport' }).toBe(
		'300px'
	);

	await setKeyboardInset(page, 0);
	await expect.poll(() => kbInset(page), { message: 'and returns to 0 when it closes' }).toBe(
		'0px'
	);
});

/**
 * Both clamps read `window.innerHeight` — the LAYOUT viewport, which the keyboard does not shrink.
 * The inset is derived from the overlay's own measured box so the test tunes itself: enough to push
 * its bottom edge under the keyboard, not so much that it no longer fits above it.
 */
async function expectsClampsAboveTheKeyboard(
	page: Page,
	openMenu: () => Promise<void>,
	menu: Locator
): Promise<void> {
	await openMenu();
	await expect(menu).toBeVisible();
	const before = (await menu.boundingBox())!;
	const innerHeight = await page.evaluate(() => window.innerHeight);
	await page.keyboard.press('Escape');
	await expect(menu).toBeHidden();

	const inset = innerHeight - (before.y + before.height) + 20;
	expect(inset, 'the overlay starts clear of the bottom edge, so an inset can reach it').toBeGreaterThan(0);

	await setKeyboardInset(page, inset);
	try {
		await openMenu();
		await expect(menu).toBeVisible();
		const box = (await menu.boundingBox())!;
		const visualHeight = innerHeight - inset;
		expect(box.y, 'the overlay stays on-screen').toBeGreaterThanOrEqual(MARGIN - 0.5);
		expect(
			box.y + box.height,
			'the overlay sits above the keyboard, not underneath it'
		).toBeLessThanOrEqual(visualHeight - MARGIN + 0.5);
		await page.keyboard.press('Escape');
		await expect(menu).toBeHidden();
	} finally {
		await setKeyboardInset(page, 0);
	}
}

test('ContextMenu places against the visual viewport, not under the keyboard', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await expectsClampsAboveTheKeyboard(
		page,
		() => page.getByTestId('topbar-save-caret').click(),
		page.locator('.context-menu').first()
	);
});

test('Popover places against the visual viewport, not under the keyboard', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	const slot = page.locator(`.slot-viewer[data-node="${uid}"]`);
	try {
		await expectsClampsAboveTheKeyboard(
			page,
			() => slot.getByTestId('viewer-settings-cog').click(),
			page.getByTestId('viewer-settings-menu')
		);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid);
	}
});
