import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { touchSession } from '../lib/touch';

/**
 * The panel header's coarse-pointer door (D-R5).
 *
 * Every structural action a panel has — Split Right, Split Down, Maximize, Change content, Close —
 * lives in one context menu opened by `oncontextmenu`. The header is `role="toolbar"
 * tabindex="-1"` with no keydown handler, so on a phone Split Right and Split Down had **no door
 * at all**. A long press is the door; the desktop right-click is untouched.
 *
 * And once the menu is open, *"Change content ▸"* still could not be expanded: its submenus were
 * `mouseenter`-only and clicking a parent row was an explicit no-op. That is part of this item.
 *
 * `maximize` is what the action half is proved with on purpose: `maximizedPanelId` is a store
 * field outside the arrangement, so it provably cannot reach the manager or the `.gfi` and this
 * spec cannot leave a layout behind for a later one (a worker's specs share one backend).
 */

const HOLD_MS = 700; // the recognizer fires at 500

function header(page: Page): Locator {
	return page.getByTestId('panel-header').first();
}

function menuRow(page: Page, label: string): Locator {
	return page
		.locator('.context-menu .item')
		.filter({ has: page.locator('.label', { hasText: new RegExp(`^${label}$`) }) });
}

/** Long-press the panel header and return the panel id it belongs to. */
async function pressHeader(page: Page): Promise<string> {
	const hdr = header(page);
	const panelId = (await hdr.evaluate(
		(el) => el.closest('.panel')!.getAttribute('data-panel-id')!
	)) as string;
	const box = (await hdr.boundingBox())!;
	// Press on the header's empty middle — not on the content button or the icon buttons, which
	// have their own actions and must not double as a menu trigger.
	const at = { x: Math.round(box.x + box.width / 2), y: Math.round(box.y + box.height / 2) };
	const touch = await touchSession(page);
	await touch.down(at);
	await page.waitForTimeout(HOLD_MS);
	await touch.up();
	return panelId;
}

test('a long press on the panel header opens the structural menu', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await pressHeader(page);

	await expect(page.locator('.context-menu').first(), 'the press opened a menu').toBeVisible();
	for (const label of ['Split Right', 'Split Down', 'Maximize', 'Change content', 'Close Panel']) {
		await expect(menuRow(page, label), label).toBeVisible();
	}
	await page.keyboard.press('Escape');
	await expect(page.locator('.context-menu')).toHaveCount(0);
});

test('the press leaves the panel active — it does not swallow setActive', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId = await pressHeader(page);
	// `Panel.svelte` marks the active panel on a CAPTURE-phase pointerdown. A recognizer that
	// stopped propagation would freeze `activePanelId` and quietly drift selection scoping.
	await expect(page.locator(`.panel.active[data-panel-id="${panelId}"]`)).toHaveCount(1);
	await page.keyboard.press('Escape');
});

test('a menu row acts: Maximize from the long-press menu maximizes the panel', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await pressHeader(page);
	await menuRow(page, 'Maximize').tap();
	await expect(page.locator('.context-menu')).toHaveCount(0);
	// The header's own control flips to Restore — the direct observable of `maximizedPanelId`.
	const restore = header(page).getByRole('button', { name: 'Restore panel' });
	await expect(restore, 'the panel is maximized').toBeVisible();

	// …and back, so the panel is handed on as it was found.
	await restore.tap();
	await expect(header(page).getByRole('button', { name: 'Maximize panel' })).toBeVisible();
});

test('“Change content ▸” expands on tap, and picking a type switches the panel', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId = await pressHeader(page);

	const parent = menuRow(page, 'Change content');
	await parent.tap();
	// The submenu is a second `.context-menu` portalled beside the first.
	await expect(page.locator('.context-menu'), 'the submenu opened on a tap').toHaveCount(2);

	await menuRow(page, 'Console').tap();
	await expect(page.locator('.context-menu')).toHaveCount(0);
	await expect(page.locator(`.panel[data-panel-id="${panelId}"]`)).toHaveAttribute(
		'data-panel-type',
		'console'
	);

	// Hand the panel back to the node editor. The type change is a command, so the canvas coming
	// back is proof the manager already holds the restore.
	await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'), panelId);
	await expect(page.locator('.canvas-wrap').first()).toBeVisible();
});
