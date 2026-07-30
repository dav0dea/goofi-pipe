import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
// The fake keyboard lives in `lib/` because `touch-expr.spec.ts` needs the same one (X's completion
// popup clamps against the visual viewport too), and a second copy is a second thing to keep true.
import { kbInset, setKeyboardInset } from '../lib/touch';

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

/**
 * The other inset the device seam owns, and the one F shipped as a no-op: the notch / rounded-corner
 * / home-indicator safe area. `viewport-fit=cover` (app.html) makes the app draw under all three, so
 * the padding is what keeps the 44px TopBar and the bottom-edge controls out from under them.
 *
 * It was stated on `body`. The shell is `position: fixed; inset: 0`, so it is laid out against the
 * INITIAL CONTAINING BLOCK — nothing on `body` can move it, and phase 1 below pins exactly that, so
 * the rule cannot drift back onto an ancestor that does not contain the app. Chromium's device
 * emulation reports `env()` as 0, which is why all four projects were green against a dead rule; the
 * insets are therefore named as tokens and stamped here, which is also how a surface that has to
 * restate them (`Toast`, itself fixed) stays in step with the shell.
 */
test('the safe-area inset is stated where the app chrome can actually feel it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const bar = page.locator('.topbar');
	const panel = page.locator('.panel').first();
	const before = (await bar.boundingBox())!;
	const panelBefore = (await panel.boundingBox())!;

	// 1. Padding on `body` cannot reach a fixed shell — the defect, made permanent as a guard.
	await page.addStyleTag({ content: 'body { padding: 44px 20px 34px 20px !important; }' });
	const onBody = (await bar.boundingBox())!;
	expect(onBody.y, 'a fixed shell is laid out against the viewport, not against body').toBe(before.y);
	expect(onBody.x, 'in both axes').toBe(before.x);

	// 2. Stated on the shell itself, every edge of the app chrome moves off the unsafe area.
	await page.evaluate(() => {
		const s = document.documentElement.style;
		s.setProperty('--safe-top', '44px');
		s.setProperty('--safe-right', '20px');
		s.setProperty('--safe-bottom', '34px');
		s.setProperty('--safe-left', '20px');
	});
	const after = (await bar.boundingBox())!;
	const panelAfter = (await panel.boundingBox())!;
	expect(after.y - before.y, 'the top bar clears the notch').toBe(44);
	expect(after.x - before.x, 'and the rounded left edge').toBe(20);
	expect(before.width - after.width, 'the bar gives up both side insets').toBe(40);
	expect(
		panelBefore.y + panelBefore.height - (panelAfter.y + panelAfter.height),
		'and the bottom-edge controls clear the home indicator'
	).toBe(34);
});
