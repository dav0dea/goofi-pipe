import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * The per-slot viewer settings cog — the second product consumer of the `Popover` primitive, and
 * the half of spec §6's "add e2e where M changes real behaviour" that was never written (the
 * ErrorPanel half was).
 *
 * Two things are pinned here. First the surface itself: the migrated menu must still render one
 * `Field` per visible setting and round-trip a change through the ViewBinding. Second — the reason
 * this file exists — its DISMISSAL. Pre-migration the menu portalled a full-screen backdrop that
 * SWALLOWED the dismissing click; the Popover migration dropped it, leaving only a bubble-phase
 * window listener, so a dismiss-click also acted on whatever was under it: it collapsed the very
 * viewer being configured (persisted, with no undo entry), or — because a slot header stops
 * pointerdown from ever reaching `window` to keep SvelteFlow from starting a node drag — toggled the
 * other slot while leaving the menu open at a now-stale anchor.
 *
 * Nodes are torn down after each test so the shared backend graph stays clean for later specs.
 */
test.describe('viewer settings menu (Popover + catcher)', () => {
	let created: string[] = [];

	test.afterEach(async ({ page }) => {
		for (const uid of created) {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
		created = [];
	});

	/** An Oscillator on the canvas; its single ARRAY output slot renders expanded by default. */
	async function oscillator(page: Page, pos: [number, number]): Promise<string> {
		const uid = await addNode(page, 'Oscillator', 'inputs', pos);
		created.push(uid);
		await waitForNode(page, uid);
		return uid;
	}

	function slot(page: Page, uid: string) {
		return page.locator(`.slot-viewer[data-node="${uid}"]`);
	}

	/** Click at a target's screen position with the RAW mouse, not `locator.click()`. Playwright's
	 * actionability check refuses to click an element another layer covers — which is the entire
	 * point here: the user aims at the header, and the catcher is what must receive the click. */
	async function clickAt(page: Page, target: ReturnType<typeof slot>): Promise<void> {
		const box = (await target.boundingBox())!;
		await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
	}

	test('the cog opens a Field per setting, and a change round-trips through the binding', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await oscillator(page, [40, 40]);
		const menu = page.getByTestId('viewer-settings-menu');

		await expect(menu, 'closed by default').toBeHidden();
		await slot(page, uid).getByTestId('viewer-settings-cog').click();
		await expect(menu, 'the cog opens the settings popover').toBeVisible();

		// The `line` schema's visible settings: Log X, Log Y, Auto, Show points (the manual Y min/max
		// are `showWhen`-gated behind Auto). Each is a Field, not a bespoke row.
		await expect(menu.locator('.ui-field'), 'one Field per visible setting').toHaveCount(4);
		const points = menu.locator('.ui-field').filter({ hasText: 'Show points' });
		const toggle = points.locator('input[type=checkbox]');
		await expect(toggle).not.toBeChecked();

		await toggle.click();
		await page.keyboard.press('Escape');
		await expect(menu).toBeHidden();

		// Re-opening re-reads the binding (the popover's content is unmounted while closed), so the
		// setting surviving the round trip is the binding having taken the write.
		await slot(page, uid).getByTestId('viewer-settings-cog').click();
		await expect(
			menu.locator('.ui-field').filter({ hasText: 'Show points' }).locator('input[type=checkbox]'),
			'the setting round-tripped through the ViewBinding'
		).toBeChecked();
		await page.keyboard.press('Escape');
		await expect(menu).toBeHidden();
	});

	test('a pointerdown on another slot dismisses the menu without toggling that slot', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const a = await oscillator(page, [40, 40]);
		const b = await oscillator(page, [40, 260]);
		const menu = page.getByTestId('viewer-settings-menu');
		const other = slot(page, b);

		await expect(other, 'the other slot starts expanded').toHaveAttribute('class', /^((?!collapsed).)*$/);
		await slot(page, a).getByTestId('viewer-settings-cog').click();
		await expect(menu).toBeVisible();

		// The other node's slot header — the exact element whose `stopPropagation` keeps the dismiss
		// pointerdown off `window`, and whose click collapses its viewer.
		await clickAt(page, other.locator('header'));
		await expect(menu, 'the click dismissed the menu').toBeHidden();
		await expect(
			other,
			'and was swallowed — the slot it landed on is untouched'
		).toHaveAttribute('class', /^((?!collapsed).)*$/);
	});

	test('a click on the configured slot dismisses the menu without collapsing it', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await oscillator(page, [40, 40]);
		const menu = page.getByTestId('viewer-settings-menu');
		const own = slot(page, uid);

		await own.getByTestId('viewer-settings-cog').click();
		await expect(menu).toBeVisible();
		await clickAt(page, own.locator('header'));
		await expect(menu, 'the click dismissed the menu').toBeHidden();
		await expect(
			own,
			'the viewer being configured is not collapsed by its own dismiss-click'
		).toHaveAttribute('class', /^((?!collapsed).)*$/);
	});
});
