import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * The workspace's own written rule (app.css: "surface steps carry separation so 1px lines
 * disappear") applied at the level that repeats most.
 *
 * `.panel` used to paint `background: var(--bg)` — byte-identical to the ground behind it — plus a
 * `1px solid var(--border)` frame, and `.panel-header` painted `--surface-1` with a second hairline
 * 26px inside that frame. So a panel was distinguishable from the workspace ONLY by lines, and every
 * split seam stacked three of them across the splitter's 8px span: panelA's border, the splitter's
 * rule, panelB's border.
 *
 * These are computed-result guards, not class assertions: the fill is read off the element and the
 * seam is counted in real composited pixels, because "how many lines does this seam paint" has no
 * DOM answer.
 */

/** Split the sole default panel to the right, through the real header context menu. */
async function splitRight(page: Page): Promise<void> {
	const header = page.getByTestId('panel-header').first();
	await header.click({ button: 'right' });
	const item = page.locator('.context-menu .item', { hasText: 'Split Right' }).first();
	await expect(item).toBeVisible();
	await item.click();
	await expect(page.locator('.panel')).toHaveCount(2);
}

test('a panel is a surface on the ground, not a rectangle drawn on it', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const surfaces = await page.locator('.panel').first().evaluate((el) => {
		const header = el.querySelector('.panel-header')!;
		const cs = getComputedStyle(el);
		return {
			panel: cs.backgroundColor,
			panelBorder: cs.borderTopColor,
			ground: getComputedStyle(document.body).backgroundColor,
			header: getComputedStyle(header).backgroundColor,
			headerBorder: getComputedStyle(header).borderBottomWidth
		};
	});

	expect(surfaces.panel, 'the panel fill is a real surface step off the ground').not.toBe(
		surfaces.ground
	);
	// The 1px box stays (it is the inset the active ring lives in — see Panel.svelte) but it paints
	// nothing: the surface step is what separates a panel from the ground, not a frame.
	expect(surfaces.panelBorder, 'and it draws no frame, because the step separates').toBe(
		'rgba(0, 0, 0, 0)'
	);
	expect(surfaces.header, 'the header is one further step, not the same surface').not.toBe(
		surfaces.panel
	);
	expect(surfaces.headerBorder, 'so it needs no hairline either').toBe('0px');
});

test('a split seam paints one neutral hairline, not a stack of them', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await splitRight(page);

	const splitter = page.locator('.splitter.row').first();
	await splitter.waitFor();
	const seam = (await splitter.boundingBox())!;

	// Scan one composited pixel row straight through the seam, 12px on either side of the splitter,
	// at the vertical middle of the panels (clear of both headers and of the flow's corner controls).
	// --border (#484848 → 72) reads far brighter than either fill it can sit on (--bg #111111 → 17,
	// --surface-1 #1c1c1c → 28), and the splitter's centered 1px rule lands on a half pixel so it
	// composites to ≈44. 35 therefore sits above every fill and below every line, antialiased or not.
	const png = (await page.screenshot()).toString('base64');
	const runs = await page.evaluate(
		async ({ png, x0, x1, y }) => {
			const img = new Image();
			img.src = `data:image/png;base64,${png}`;
			await img.decode();
			const scale = img.width / window.innerWidth;
			const canvas = document.createElement('canvas');
			canvas.width = img.width;
			canvas.height = img.height;
			const ctx = canvas.getContext('2d')!;
			ctx.drawImage(img, 0, 0);
			const { data } = ctx.getImageData(
				Math.round(x0 * scale),
				Math.round(y * scale),
				Math.round((x1 - x0) * scale),
				1
			);
			// Count RUNS of bright NEUTRAL columns, so an antialiased line still counts once and the
			// newly-active panel's accent ring — which is state, not chrome — is not counted as one.
			// A --border grey is r≈g≈b; --ring-accent over any fill here leads green by ~58.
			let n = 0;
			let inRun = false;
			for (let i = 0; i < data.length; i += 4) {
				const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
				const line = Math.max(r, g, b) >= 35 && g - r < 20;
				if (line && !inRun) n++;
				inRun = line;
			}
			return n;
		},
		{ png, x0: seam.x - 12, x1: seam.x + seam.width + 12, y: seam.y + seam.height / 2 }
	);

	// Exactly one: the splitter's own rule. (Before, this scan read the two panel borders as well —
	// three lines across an 8px span, the third of them tinted by the active panel's ring.)
	expect(runs, 'the seam reads through the splitter rule alone').toBe(1);
});
