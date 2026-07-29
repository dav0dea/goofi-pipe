import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * R Task 5 — a CANCELLED drag must leave nothing behind (§3.1g), and a panel has a PIXEL floor
 * (D-R10). Both are desktop bugs, which is why they are pinned in the fine-pointer project: the
 * corner grip is a fine-pointer gesture by R-Task 3's decision, and a narrow desktop window
 * collapses a panel exactly the way a phone does.
 *
 * The corner grip armed `pointermove`/`pointerup` on the window and detached on `pointerup` ALONE.
 * A pointer can be CANCELLED instead of released — the browser reclaims the gesture for a pan, a
 * system UI takes over, the owning panel unmounts mid-drag — and then the listeners survived, the
 * preview ghost stayed painted, and the NEXT release anywhere in the app committed the abandoned
 * intent: a `ws.split()` or a `ws.close()` restructuring the workspace one click later (H6).
 *
 * The cancel is dispatched rather than provoked, because provoking it needs a browser decision this
 * harness cannot make. `ui/dragGesture.ts` owns the state machine (unit-tested); this proves the
 * real grip is wired to it.
 */

const panels = (page: Page) => page.locator('[data-panel-id]');

test('a cancelled corner drag drops its ghost and never commits a split', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const before = await panels(page).count();

	// The grip is a 16px box clipped to the panel body's top-left triangle.
	const body = (await page.locator('.panel-body').first().boundingBox())!;
	const grip = { x: Math.round(body.x + 3), y: Math.round(body.y + 3) };

	try {
		await page.mouse.move(grip.x, grip.y);
		await page.mouse.down();
		// Past THRESHOLD (24px), which is what arms a split intent and paints the preview.
		await page.mouse.move(grip.x + 160, grip.y + 20);
		await expect(page.locator('.drag-ghost'), 'the drag is really in flight').toHaveCount(1);

		await page.locator('.panel-body .corner.tl').first().dispatchEvent('pointercancel');
		await expect(
			page.locator('.drag-ghost'),
			'a cancelled gesture stops previewing a result it will not produce'
		).toHaveCount(0);

		// The stale-commit half: the release used to run the leaked window `pointerup`.
		await page.mouse.up();
		await page.waitForTimeout(150);
		expect(await panels(page).count(), 'and it commits nothing when the button comes up').toBe(
			before
		);
	} finally {
		// If the leak fired, the split is a tracked layout command — undo it, so the shared workspace
		// (and the 400ms layout push that outlives this page) is handed back as it was found.
		for (let i = 0; i < 4 && (await panels(page).count()) > before; i++) {
			await page.evaluate(() => (window as any).goofi.commands.undo());
			await page.waitForTimeout(150);
		}
		await page.waitForTimeout(700);
	}
});

test('a panel cannot be dragged below its pixel floor', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const before = await panels(page).count();

	// Split through the header menu — the same route `panel-surface.spec.ts` uses.
	await page.getByTestId('panel-header').first().click({ button: 'right' });
	await page.locator('.context-menu .item', { hasText: 'Split Right' }).first().click();
	await expect(panels(page)).toHaveCount(before + 1);

	try {
		const seam = (await page.locator('.splitter.row').first().boundingBox())!;
		const at = { x: Math.round(seam.x + seam.width / 2), y: Math.round(seam.y + seam.height / 2) };

		// Drag the seam hard into the left wall. MIN_FRACTION alone is 5% of whatever the window
		// happens to be — 64px here, and a 20px sliver on a 412px phone. MIN_PANEL_PX is what stops
		// it at something that is still recognisably a panel.
		await page.mouse.move(at.x, at.y);
		await page.mouse.down();
		for (let x = at.x; x > 0; x -= 60) await page.mouse.move(x, at.y);
		await page.mouse.move(-500, at.y);
		await page.mouse.up();

		const w = await panels(page)
			.first()
			.evaluate((el) => el.getBoundingClientRect().width);
		expect(
			w,
			'the floor is a size, not a percentage of whatever the window happens to be'
		).toBeGreaterThanOrEqual(115);
	} finally {
		await page
			.getByTestId('panel-header')
			.nth(1)
			.getByRole('button', { name: 'Close panel' })
			.click();
		await expect(panels(page), 'the workspace is back to one panel').toHaveCount(before);
		await page.waitForTimeout(700);
	}
});
