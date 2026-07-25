import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

// The workspace panel system is FROZEN UX; sub-project M restyled its chrome onto the
// `$lib/ui` primitives (PanelHeader's dropdown + maximize/close, the tab strip's ✕/＋).
// Nothing else in the suite exercises that chrome, so these are the regression guards for
// the two invariants a primitive swap can silently break: the header dropdown's ContextMenu
// wiring, and the tab strip's zero-width collapsed close.

test('the panel header dropdown opens the context menu and Escape dismisses it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const header = page.getByTestId('panel-header').first();
	await header.waitFor();
	const menu = page.locator('.context-menu');

	await header.locator('.content-btn').click();
	await expect(menu, 'the content dropdown opens a context menu').toHaveCount(1);
	await expect(menu.locator('.item').first()).toBeVisible();

	await page.keyboard.press('Escape');
	await expect(menu, 'Escape dismisses it').toHaveCount(0);
});

test('a collapsed tab close button occupies zero width (the even-padding invariant)', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// A close ✕ only renders once there is more than one tab.
	const tabs = page.getByTestId('workspace-tabs');
	await tabs.getByRole('button', { name: 'New tab' }).click();
	const close = tabs.getByRole('button', { name: 'Close tab' }).first();
	await close.waitFor({ state: 'attached' });

	// Its tab is neither hovered nor active, so the ✕ is collapsed: it must take NO
	// horizontal space at all, or every inactive tab is padded wider than its neighbours.
	// (A primitive with a 1px border clamps `width: 0` to 2px under border-box — the exact
	// regression this guards.)
	await expect
		.poll(async () => (await close.boundingBox())?.width, {
			message: 'the collapsed ✕ takes zero width'
		})
		.toBe(0);

	// Hovering its tab reveals it at the frozen 16px reveal width.
	await close.locator('xpath=..').hover();
	await expect.poll(async () => (await close.boundingBox())?.width).toBe(16);
});

test('the tab strip ＋ keeps its frozen 22px box (not the primitive --hit floor)', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// IconButton floors its box to --hit (28px on a fine pointer). The tab pills are ~23px
	// tall, so an unpinned ＋ would stand visibly taller than the tabs beside it — the tab
	// strip is frozen geometry and pins it back to 22, exactly as `.close` pins 16.
	const add = page.getByTestId('workspace-tabs').getByRole('button', { name: 'New tab' });
	await add.waitFor();
	const box = (await add.boundingBox())!;
	expect(box.width, 'the ＋ keeps its pre-migration 22px width').toBe(22);
	expect(box.height, 'the ＋ keeps its pre-migration 22px height').toBe(22);
});
