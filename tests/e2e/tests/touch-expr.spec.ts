import { test, expect, type Locator, type Page } from '@playwright/test';
import { setKeyboardInset } from '../lib/touch';

/**
 * The coarse half of the expression editor (D-X8). Runs ONLY under the `touch` project (Pixel 7 →
 * `(pointer: coarse)` / `(hover: none)` true, real touch input available), because every claim here is
 * about a device the `default` project is not.
 *
 * Three things touch needs that a desktop editor gets for free, and none of them are inherited: the
 * editable element is a contenteditable `<div>`, so `app.css`'s `input, select, textarea` coarse floors
 * — both the 16px focus-zoom one and the `--hit` one — cannot reach it; and the completion popup is
 * placed by CodeMirror, whose own default measures `window.innerHeight`, which a soft keyboard does not
 * shrink.
 *
 * Driven on `/dev/inspector`, whose fx sample needs no live graph: what is under test is the editor's
 * touch behaviour, and the four names the evaluator injects (`nd`, `t`, `np`, `globals`) are offered
 * with no patch at all — through the same merged popup as Python's own builtins.
 */

const MARGIN = 6; // the popup's viewport-edge margin, matching clampToViewport's

/** The gallery's fx field, switched into expression mode; returns its inline editor. */
async function fxEditor(page: Page): Promise<Locator> {
	await page.goto('/dev/inspector');
	const field = page.getByTestId('inspector-fx');
	await field.getByTestId('param-fx-toggle').click();
	const editor = field.getByTestId('param-expr-input');
	await expect(editor).toBeVisible();
	return editor;
}

const popup = (page: Page) => page.locator('.cm-tooltip-autocomplete');

/** Type into the editor from scratch and settle past the completion debounce. */
async function retype(page: Page, editor: Locator, src: string): Promise<void> {
	await editor.tap();
	await page.keyboard.press('Control+a');
	await page.keyboard.press('Delete');
	await page.keyboard.type(src, { delay: 10 });
	await page.waitForTimeout(300);
}

test('the inline expression editor is >= 16px under a coarse pointer', async ({ page }) => {
	const editor = await fxEditor(page);
	// `.cm-content` inherits from the host, which is where the floor is stated — so this measures the
	// element that actually takes focus, the same way `touch-inspector.spec.ts` measures the expanded one.
	const px = await editor.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
	expect(px, 'below 16px iOS force-zooms the viewport on focus').toBeGreaterThanOrEqual(16);
});

test('a completion accepts by TAP, and its rows are real tap targets', async ({ page }) => {
	const editor = await fxEditor(page);
	await retype(page, editor, 'n');
	await expect(popup(page), 'the merged popup opened').toHaveCount(1);
	const row = popup(page).locator('li', { hasText: 'nd' }).first();
	await expect(row).toBeVisible();
	const box = (await row.boundingBox())!;
	const hit = await page.evaluate(() =>
		parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hit'))
	);
	expect(box.height, 'a completion row is as tappable as anything else').toBeGreaterThanOrEqual(hit);

	await row.tap();
	await expect(editor, 'the tap accepted the completion').toHaveText('nd');
});

/**
 * D-X8's popup clause. The tooltip is parented to `document.body` and constrained through CodeMirror's
 * `tooltipSpace` hook by `overlayViewport()` — the app's one soft-keyboard-aware measurement, the same
 * one `Popover` and `ContextMenu` clamp against. CodeMirror's own default reads `window.innerHeight`,
 * which a keyboard does not shrink, so without the hook the list is parked underneath the very keyboard
 * that raised it.
 *
 * The inset is derived from the popup's own measured box, so the test tunes itself: enough to push its
 * bottom edge under the keyboard, not so much that it no longer fits above it.
 */
test('the completion popup stays clear of the soft keyboard', async ({ page }) => {
	const editor = await fxEditor(page);
	await retype(page, editor, 'n');
	await expect(popup(page)).toBeVisible();
	const before = (await popup(page).boundingBox())!;
	const innerHeight = await page.evaluate(() => window.innerHeight);
	const inset = innerHeight - (before.y + before.height) + 20;
	expect(inset, 'the popup starts clear of the bottom edge, so an inset can reach it').toBeGreaterThan(0);

	try {
		// The keyboard is up BEFORE the list opens, which is the real order: you type with it showing.
		await setKeyboardInset(page, inset);
		await retype(page, editor, 'np');
		await expect(popup(page)).toBeVisible();
		const box = (await popup(page).boundingBox())!;
		expect(box.y, 'the popup stays on-screen').toBeGreaterThanOrEqual(MARGIN - 0.5);
		expect(
			box.y + box.height,
			'the popup sits above the keyboard, not underneath it'
		).toBeLessThanOrEqual(innerHeight - inset - MARGIN + 0.5);
	} finally {
		await setKeyboardInset(page, 0);
	}
});
