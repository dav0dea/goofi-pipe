import { test, expect, type Page } from '@playwright/test';
import { closeSplit, splitRight, waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

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
	try {
		await seamRunsAreOne(page);
	} finally {
		await closeSplit(page);
	}
});

/** The pixel readback, factored out so the split above can be undone in a `finally`. */
async function seamRunsAreOne(page: Page): Promise<void> {
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
}

/** The mean channel value the compositor actually produced over a CSS-px rect, 0–255. */
async function meanPixel(
	page: Page,
	rect: { x: number; y: number; width: number; height: number }
): Promise<number> {
	const png = (await page.screenshot()).toString('base64');
	return page.evaluate(
		async ({ png, r }) => {
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
				Math.round(r.x * scale),
				Math.round(r.y * scale),
				Math.max(1, Math.round(r.width * scale)),
				Math.max(1, Math.round(r.height * scale))
			);
			let sum = 0;
			for (let i = 0; i < data.length; i += 4) sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
			return sum / (data.length / 4);
		},
		{ png, r: rect }
	);
}

/** The id of the sole default panel. */
async function solePanelId(page: Page): Promise<string> {
	const panels = await page.evaluate(() => (window as any).goofi.query.panels());
	return panels[0].panelId;
}

/** Hand the panel back to the node editor, past the layout debounce (see `closeSplit`). */
async function restoreEditor(page: Page, panelId: string): Promise<void> {
	await page.evaluate(
		(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
		panelId
	);
	await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
	await page.waitForTimeout(700); // past AppShell's 400ms set_layout debounce
}

/** The largest (green − red) the compositor produced over a CSS-px rect. --accent (#50d0a0) at
 * 45% over any neutral in the chrome leads green by ~58; every neutral leads by 0. */
async function greenLead(
	page: Page,
	rect: { x: number; y: number; width: number; height: number }
): Promise<number> {
	const png = (await page.screenshot()).toString('base64');
	return page.evaluate(
		async ({ png, r }) => {
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
				Math.round(r.x * scale),
				Math.round(r.y * scale),
				Math.max(1, Math.round(r.width * scale)),
				Math.max(1, Math.round(r.height * scale))
			);
			let lead = 0;
			for (let i = 0; i < data.length; i += 4) lead = Math.max(lead, data[i + 1] - data[i]);
			return lead;
		},
		{ png, r: rect }
	);
}

/**
 * The active-panel ring is ONE mechanism, owned by `Panel`, and it frames the WHOLE panel —
 * header included — whatever the content happens to be.
 *
 * Four panel types used to opt out of it (a `contentOutline` flag on the registry entry) and draw
 * their own ring around just their body instead, so on Viewer/Parameters/Metadata/Globals the panel
 * header sat OUTSIDE the focus indication while on the Node Editor/Console/Empty it sat inside it —
 * the same state drawn two different shapes depending on which panel had focus. Only a pixel
 * readback can tell them apart: both spellings put an `active` class on a live rule.
 */
const PANEL_TYPES = ['node-editor', 'console', 'viewer', 'parameters', 'metadata', 'globals', 'empty'];

test('every panel type rings the whole panel, its header included', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId = await solePanelId(page);
	const panel = page.locator('.panel').first();
	try {
		for (const type of PANEL_TYPES) {
			await page.evaluate(
				([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
				[panelId, type] as const
			);
			await expect(panel).toHaveAttribute('data-panel-type', type);
			await expect(panel, 'the sole panel is the active one').toHaveClass(/\bactive\b/);
			const box = (await panel.boundingBox())!;
			const headerH = await panel
				.locator('.panel-header')
				.evaluate((el) => el.getBoundingClientRect().height);

			// Beside the header: a 3px band down the panel's left edge, spanning the header's height.
			const beside = await greenLead(page, {
				x: box.x,
				y: box.y + 2,
				width: 3,
				height: headerH - 4
			});
			expect(beside, `${type}: the ring runs down the panel edge beside its header`).toBeGreaterThan(
				20
			);

			// And above it: the panel's own top edge, sampled mid-width clear of the corner radius.
			const above = await greenLead(page, {
				x: box.x + box.width / 2 - 20,
				y: box.y,
				width: 40,
				height: 2
			});
			expect(above, `${type}: the ring closes above its header`).toBeGreaterThan(20);
		}
	} finally {
		await restoreEditor(page, panelId);
	}
});

/**
 * The boundary the first test above is blind to. M-10 deleted `.panel-header`'s hairline AND
 * promoted `Bar` to the header's own `--surface-2`, so on the four panel types that render a `Bar`
 * flush at the top of the panel body the two strips merged into one undelimited ~92px chrome slab:
 * no line, and no step either. D5 says a surface step carries the separation *instead of* a border —
 * deleting the border with nothing behind it is not a saliency win, it is a lost boundary.
 *
 * Asserted on the RESULT, not on which token supplies it: whatever the ladder does, these two
 * adjacent strips must differ, and neither may buy that back with a hairline.
 */
test('a content toolbar flush under the panel header is a step off it, not one slab', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId = await solePanelId(page);
	try {
		await page.evaluate(
			(id) => (window as any).goofi.commands.setPanelType(id, 'console'),
			panelId
		);
		await expect(page.getByTestId('console-panel')).toBeVisible();

		const m = await page.locator('.panel').first().evaluate((el) => {
			const cs = (n: Element) => getComputedStyle(n);
			const header = el.querySelector('.panel-header')!;
			const body = el.querySelector('.panel-body')!;
			const bar = el.querySelector('.panel-body .ui-bar')!;
			return {
				header: cs(header).backgroundColor,
				headerBorder: cs(header).borderBottomWidth,
				bar: cs(bar).backgroundColor,
				barBorder: cs(bar).borderTopWidth,
				flush: bar.getBoundingClientRect().top - body.getBoundingClientRect().top
			};
		});

		expect(m.flush, 'the toolbar really does sit flush under the header').toBeLessThanOrEqual(1);
		expect(m.bar, 'so the two strips must be a real surface step apart').not.toBe(m.header);
		expect(m.headerBorder, 'and the step is what separates them, not a restored hairline').toBe(
			'0px'
		);
		expect(m.barBorder, 'from either side').toBe('0px');
	} finally {
		await restoreEditor(page, panelId);
	}
});

/**
 * The same promotion's other half. `IconButton`'s ghost hover fill was the ABSOLUTE `--surface-2` —
 * which is exactly the strip a ghost in chrome sits on once `Bar` and `.panel-header` took that
 * rung — so hovering one painted nothing at all. A ghost control has no surface of its own, so its
 * hover has to LIFT whatever it happens to sit on; naming a rung can only ever be right on one.
 *
 * The reachable, uncompensated instance is `NodeLinkedPanel`'s unlink ✕ (the panel header's own
 * buttons at least brighten their ink). Only a pixel readback can answer "did anything appear" —
 * a computed `background-color` reads back the declared value whether or not it is visible.
 */
test('a ghost icon button on a content toolbar paints a visible hover fill', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const panelId = await solePanelId(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	try {
		// The link is stored as the node UID (stable across rename), not its display name.
		await page.evaluate(
			([id, u]) => {
				(window as any).goofi.commands.setPanelType(id, 'parameters');
				(window as any).goofi.commands.bindNodeToPanel(id, u);
			},
			[panelId, uid] as const
		);

		const btn = page.getByTestId('node-linked-panel').getByRole('button', { name: 'Unlink node' });
		await expect(btn).toBeVisible();
		const box = (await btn.boundingBox())!;
		// A 4px corner square, clear of the centred glyph, so this reads the FILL and nothing else.
		const corner = { x: box.x + 1, y: box.y + 1, width: 4, height: 4 };

		await page.mouse.move(5, 5);
		await page.waitForTimeout(200); // past the --dur-fast fill transition
		const rest = await meanPixel(page, corner);
		await btn.hover();
		await page.waitForTimeout(200);
		const hovered = await meanPixel(page, corner);

		expect(
			hovered - rest,
			'the ghost hover fill lifts the toolbar it sits on, instead of repainting it'
		).toBeGreaterThan(5);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
		await restoreEditor(page, panelId);
	}
});
