import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addGlobal, addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * R Task 5 (§3.1e/f) — what a narrow viewport clips today.
 *
 * Each of these surfaces has a fixed cost it never gives up, so the thing at the END of the row is
 * the thing that disappears: the file browser's Save button, the console's copy button, the globals
 * table's inputs. "Off the right edge with no scroll recovery" is not a small-screen inconvenience,
 * it is the control not existing.
 *
 * Measured against the CONTAINER, never a literal, so the assertions still mean something at any
 * width and cannot be greened by a lucky viewport.
 */

/** How far `inner` overflows `outer` horizontally, in px (0 = fully inside). */
async function overflowRight(inner: Locator, outer: Locator): Promise<number> {
	const i = (await inner.boundingBox())!;
	const o = (await outer.boundingBox())!;
	expect(i, 'the control is rendered at all').toBeTruthy();
	return Math.max(0, i.x + i.width - (o.x + o.width));
}

/** Borrow the first panel for `type`, run `body`, and hand the workspace back. */
async function withPanelType(page: Page, type: string, body: () => Promise<void>): Promise<void> {
	const panelId: string = await page.evaluate(
		() => (window as any).goofi.query.panels()[0].panelId
	);
	await page.evaluate(
		([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
		[panelId, type] as const
	);
	try {
		await body();
	} finally {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
			panelId
		);
		await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		// AppShell's layout push is debounced 400ms and outlives the page.
		await page.waitForTimeout(700);
	}
}

/* 320px — the narrowest phone still in real use, and the width at which every fixed cost in a row
   stops fitting. The Pixel-7 project's own 412px is where these surfaces got tight; this is where
   they broke. `hasTouch`/`isMobile` carry over, so the coarse floors are still in force. */
test.describe('at 320px', () => {
	test.use({ viewport: { width: 320, height: 690 } });

	test('the file browser keeps its Save button on screen', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// Ctrl+S, not the TopBar button: the header's own actions are still off-screen here (that is
		// Task 6's bug, not this one). A fresh page has no save path, so Save opens the browser.
		await page.keyboard.press('Control+s');
		const modal = page.getByTestId('fs-browser');
		await expect(modal).toBeVisible();
		try {
			expect(
				await overflowRight(page.getByTestId('fs-save'), modal),
				'the Save button is inside the modal'
			).toBe(0);
			expect(
				await overflowRight(page.getByTestId('fs-filename'), modal),
				'and so is the name it saves under'
			).toBe(0);

			// The root sidebar was non-shrinking, so the file list got whatever was left of a modal
			// that is itself only 92vw. Compared against the modal, so this holds at any width.
			const list = (await page.getByTestId('fs-list').boundingBox())!;
			const m = (await modal.boundingBox())!;
			expect(list.width, 'the file list gets the majority of a narrow modal').toBeGreaterThan(
				m.width * 0.6
			);
		} finally {
			await page.keyboard.press('Escape');
			await expect(modal).toHaveCount(0);
		}
	});

	test('the globals table keeps its inputs usable and its delete reachable', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const name = `narrow_${Date.now()}`;
		await addGlobal(page, name, 1, 'float');
		await withPanelType(page, 'globals', async () => {
			try {
				const row = page.locator(`[data-testid="global-row"][data-name="${name}"]`);
				await expect(row).toBeVisible();
				expect(
					await overflowRight(row.getByTestId('global-delete'), row),
					'the delete control is inside its row'
				).toBe(0);

				const r = (await row.boundingBox())!;
				const nameBox = (await row.getByTestId('global-name').boundingBox())!;
				// The actions column claimed a flat 15% but never rendered under ~85px, so it stole
				// from the two editable columns at every width below ~565px.
				expect(
					nameBox.width,
					'the name is still editable, not squeezed to a sliver'
				).toBeGreaterThan(r.width * 0.3);
			} finally {
				await page.evaluate((n) => (window as any).goofi.commands.removeGlobal(n), name);
			}
		});
	});

	test('a console row keeps its copy button inside the scroller', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await withPanelType(page, 'console', async () => {
			const uid = await addNode(page, 'LempelZiv', 'python');
			try {
				await waitForNode(page, uid);
				await expect(page.getByTestId('console-entry').first()).toBeVisible();
				// `.scroll` clips horizontally, so anything past its right edge is simply gone — the
				// row's ~258px of fixed cost is what pushed the copy button out.
				expect(
					await overflowRight(page.getByTestId('console-copy').first(), page.locator('.scroll')),
					'the copy button is inside the scroller that clips it'
				).toBe(0);
			} finally {
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});
	});
});

test('the inspector leaves the canvas it overlays reachable', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	try {
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		const pane = page.getByTestId('auto-side-panel');
		await expect(pane).toBeVisible();
		const p = (await pane.boundingBox())!;
		const host = (await page.locator('.editor-panel').first().boundingBox())!;
		// Its width is JS-inline (a stored 420px default) and was clamped only to [260, 720] — never
		// to the host — so on a ~386px editor it covered the canvas completely with its own left edge
		// clipped off. Deselecting by tapping the canvas is the ONLY way to close it, so covering the
		// canvas is a dead end.
		expect(p.width, 'the pane leaves a strip of canvas').toBeLessThan(host.width);
		expect(
			host.x + host.width - (p.x + p.width),
			'and is not clipped at its own edge'
		).toBeLessThan(2);
		expect(p.x - host.x, 'the strip is at least one tap target wide').toBeGreaterThanOrEqual(44);
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});

test('the add-node menu fits the screen it opens on, keyboard and all', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
	const menu = page.getByTestId('add-node-menu-anchor');
	await expect(menu).toBeVisible();
	try {
		const box = (await menu.boundingBox())!;
		const vp = page.viewportSize()!;
		expect(box.x).toBeGreaterThanOrEqual(0);
		expect(box.x + box.width, 'the menu fits the viewport it opened on').toBeLessThanOrEqual(
			vp.width
		);
		expect(box.y + box.height, 'vertically too').toBeLessThanOrEqual(vp.height);

		// Its search input declared `font-size: var(--fs-body)` as a scoped `input` rule, which
		// out-specifies app.css's coarse 16px floor (0,0,1) — so focusing it force-zooms iOS.
		const fs = await page
			.getByTestId('add-menu-search')
			.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
		expect(fs, 'and focusing its search does not force-zoom iOS').toBeGreaterThanOrEqual(16);
	} finally {
		await page.keyboard.press('Escape');
		await expect(menu).toHaveCount(0);
	}
});
