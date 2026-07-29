import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { AS_ROWS, PRIORITY, inBar, menuRow, openOverflow } from '../lib/topbar';
import { emptySpot } from '../lib/touch';

/**
 * What actually reflows — run at every coarse geometry R claims to support.
 *
 * `playwright.config.ts` points three projects at this ONE file: `touch` (Pixel 7 portrait,
 * 412×839), `touch-landscape` (863×360) and `tablet` (712×1138). Everything else under `touch-*`
 * stays in the portrait project alone, on purpose — the hit floors, the hover doors and the
 * long-press doors are all driven by `@media (hover: none) and (pointer: coarse)`, which answers
 * the same at 412px and at 1138px, so re-running them would triple the wall clock to re-measure a
 * constant. What is NOT constant is anything that fits or does not fit:
 *
 *  - the header's progressive overflow, whose budget is measured from INTRINSIC widths — and
 *    those widths are larger under the coarse `--hit` floor than at any width the desktop spec
 *    can reach by resizing, so this is not `topbar-overflow.spec.ts` at another number;
 *  - the inspector's clamp against its host, which is a fraction of the editor, not of the phone;
 *  - the add-node menu's clamp against the viewport, in both orientations;
 *  - and whether a 360px-TALL viewport leaves a canvas to author on at all.
 *
 * Hermeticity: nothing here borrows a panel or a layout tab; the one test that adds a node removes
 * it, and the one that names the patch hands the backend back unnamed. Three projects run this
 * file against the SAME backend, one after another, so each has to leave it as it found it.
 */

const hit = (page: Page): Promise<number> =>
	page.evaluate(() =>
		parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--hit'))
	);

/** How far `inner` falls outside `outer`, on any side, in px (0 = fully inside). */
function outside(
	inner: { x: number; y: number; width: number; height: number },
	outer: { x: number; y: number; width: number; height: number }
): number {
	return Math.max(
		0,
		outer.x - inner.x,
		outer.y - inner.y,
		inner.x + inner.width - (outer.x + outer.width),
		inner.y + inner.height - (outer.y + outer.height)
	);
}

/**
 * A patch name long enough to crowd the header, and a scratch directory to keep it in.
 *
 * The name is not decoration. R-Task 6's after-screenshot caught what no measurement had: on a
 * LIVE patch the widest thing in the bar is the filename, and `.brand` was `flex: 0 0 auto`, so at
 * 412px the brand claimed ~375 of 391px and pushed the overflow trigger — the one control that
 * must always be reachable — clean off the right edge. An unnamed empty patch cannot reproduce
 * that, so a guard written against the boot state would be green on the very defect it exists to
 * catch. Verified by mutation: with `.brand` put back to `flex: 0 0 auto`, this test fails at
 * 412px and at 712px on "the overflow trigger is inside the bar" (the brand measured 547.9px wide
 * in a 412px bar, the trigger's left edge at x=681) — and only because the patch is named.
 */
const CROWDING_NAME = 'a-patch-with-a-deliberately-long-name-that-crowds-the-header';
let scratch = '';

test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-reflow-')));
});
test.afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }));

/**
 * Save the patch under `CROWDING_NAME`, through the real Save flow.
 *
 * Not `window.goofi.commands.save(path)`, which is the same RPC but leaves the header unnamed:
 * the `save` arm broadcasts no `save_path_changed` (only `load` does), so it is `AppShell` that
 * publishes the new path locally after ITS save — and the façade calls the store directly, below
 * that. So the only way to reach the state this test is about is the flow a user takes.
 *
 * `topbar-save` is dispatched rather than tapped because at 412px it has spilled out of the bar
 * and is `display: none`; reaching it through the overflow menu is `topbar-overflow.spec.ts`'s
 * question, not this file's.
 */
async function saveAsCrowdingName(page: Page): Promise<void> {
	await page.getByTestId('topbar-save').evaluate((el: HTMLElement) => el.click());
	const modal = page.getByTestId('fs-browser');
	await expect(modal).toBeVisible();
	const bar = modal.getByTestId('fs-path-input');
	await bar.fill(scratch);
	await bar.press('Enter');
	await expect(bar).toHaveValue(scratch);
	await modal.getByTestId('fs-filename').fill(CROWDING_NAME);
	await modal.getByTestId('fs-save').click();
	await expect(modal, 'confirming Save closes the browser').toBeHidden();
	await expect(page.locator('.topbar .path'), 'the header is showing the patch name').toHaveText(
		`${CROWDING_NAME}.gfi`
	);
}

test('the header keeps every action reachable, in the bar or in the menu', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await saveAsCrowdingName(page);
	try {
		await checkHeaderFits(page);
	} finally {
		// Hand the backend back unnamed: `loadText` (a load with no path) is what resets `save_path`
		// to null, which later specs assume when they click Save expecting the browser.
		await page.evaluate(
			(y) => (window as any).goofi.commands.loadText(y),
			fs.readFileSync(path.join(scratch, `${CROWDING_NAME}.gfi`), 'utf8')
		);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.toBe(null);
	}
});

/**
 * The header's action list, once its ResizeObserver-driven re-plan has stopped moving.
 *
 * Naming the patch changes the brand's width, which re-fires the observer that decides what
 * spills — so a read taken the instant the Save dialog closes can catch a plan mid-flight and
 * report an action as "in the bar" one frame before it is `display: none`. Two agreeing reads is
 * also the assertion that it settles at all, which is trap 1 (oscillation) seen from the outside.
 */
async function settledBar(page: Page): Promise<string[]> {
	let prev = (await inBar(page)).join();
	for (let i = 0; i < 20; i++) {
		await page.waitForTimeout(50);
		const now = (await inBar(page)).join();
		if (now === prev) return prev === '' ? [] : prev.split(',');
		prev = now;
	}
	throw new Error(`the header's overflow plan never settled (last: ${prev})`);
}

async function checkHeaderFits(page: Page): Promise<void> {
	const kept = await settledBar(page);
	const bar = (await page.locator('.topbar').boundingBox())!;

	// Whatever the bar kept, it kept INSIDE itself. The failure this guards is the one the
	// after-screenshot caught in R-Task 6: the actions were rendered and simply off the right edge,
	// which no visibility assertion can see.
	for (const id of kept) {
		const box = await page.getByTestId(id).boundingBox();
		expect(box, `${id} is in the bar (kept: ${kept.join()}) and has a box`).not.toBeNull();
		expect(outside(box!, bar), `${id} is inside the bar`).toBeLessThanOrEqual(1);
	}

	// The trigger is resident at every width and is the only door to whatever spilled, so it is the
	// one control that must never be the thing pushed off the edge.
	const trigger = page.getByTestId('topbar-overflow');
	await expect(trigger).toBeVisible();
	const t = (await trigger.boundingBox())!;
	expect(
		outside(t, bar),
		`the overflow trigger is inside the bar (trigger ${JSON.stringify(t)} in ${JSON.stringify(bar)})`
	).toBeLessThanOrEqual(1);

	// The tab strip is what the actions spill in favour of; squeezed to zero, no layout tab and not
	// even the ＋ that makes one can be reached (o1). TopBar reserves `2 × --hit` for it in the
	// overflow BUDGET — but the brand competes for the same slack in the layout that follows, so
	// with a long filename the strip lands under that reservation at the narrow end: measured
	// 55.5px at 412px, against 95.7px at 712px and 130.9px at 863px. Left for R's audit rather
	// than papered over here — the strip is its own `overflow-x` scroller, so what does not fit is
	// scrolled to rather than lost. A tap target wide is what it must never go below, which is the
	// floor's purpose rather than its arithmetic.
	const slot = (await page.locator('.topbar .tabslot').boundingBox())!;
	expect(slot.width, 'the tab strip is at least a tap target wide').toBeGreaterThanOrEqual(
		await hit(page)
	);
	await expect(page.getByRole('button', { name: 'New tab' })).toBeVisible();

	// And nothing was simply lost between the two representations.
	await openOverflow(page);
	try {
		for (const id of PRIORITY) {
			const spilled = !kept.includes(id);
			for (const label of AS_ROWS[id]) {
				await expect(menuRow(page, label), `${label} (${id} spilled: ${spilled})`).toHaveCount(
					spilled ? 1 : 0
				);
			}
		}

		// …and nothing was lost BELOW THE FOLD either. `toHaveCount` is green for a row rendered
		// 250px under the bottom of the screen, which is what a 604px menu is on a 360px landscape
		// phone: the surface had no `max-height` and no scroller, and `clampToViewport` FLOORS an
		// oversized menu at its 6px margin rather than fitting it. Those rows are the canvas
		// commands (D-R4) — the only pointer door there is to Delete / Group / Copy / Paste /
		// Duplicate — so off the fold is the same as absent.
		const vp = page.viewportSize()!;
		const screen = { x: 0, y: 0, width: vp.width, height: vp.height };
		const menu = page.locator('.context-menu').first();
		expect(
			outside((await menu.boundingBox())!, screen),
			'the overflow menu fits the screen it opened on'
		).toBeLessThanOrEqual(1);
		const rows = menu.locator('.item');
		const n = await rows.count();
		for (let i = 0; i < n; i++) {
			const row = rows.nth(i);
			// Scrolled into view first: once the surface bounds itself, the lower rows are legitimately
			// off-screen until the menu is scrolled, and REACHABLE is the property under test.
			await row.scrollIntoViewIfNeeded();
			expect(
				outside((await row.boundingBox())!, screen),
				`the "${await row.locator('.label').innerText()}" row is reachable on screen`
			).toBeLessThanOrEqual(1);
		}
	} finally {
		await page.keyboard.press('Escape');
	}
}

/* Moved here from `touch-narrow.spec.ts` (which keeps the fixed-cost rows it was written for):
   the pane's width is JS-inline and clamped against its HOST, so what it leaves behind is a
   fraction of the editor panel — a different answer in portrait, in landscape and on a tablet. */
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
		// Before R its clamp was [260, 720] and never the host, so on a ~386px editor it covered the
		// canvas completely with its own left edge clipped off. Deselecting by tapping the canvas is
		// the only way to close it, so covering the canvas is a dead end.
		expect(p.width, 'the pane leaves a strip of canvas').toBeLessThan(host.width);
		expect(
			host.x + host.width - (p.x + p.width),
			'and is not clipped at its own edge'
		).toBeLessThan(2);
		expect(p.x - host.x, 'the strip is at least one tap target wide').toBeGreaterThanOrEqual(
			await hit(page)
		);
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});

/* Also moved: the add-node menu is the one popover that opens at a point rather than off an
   anchor, so which edge it has to be pushed away from is entirely the viewport's shape. */
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

/**
 * The other axis, and the one only a landscape phone asks: 360px of height, minus the app header
 * and the panel header, still has to leave something to author in. This is what the `vh` → `dvh`
 * sweep bought — a `100vh` shell on a phone is TALLER than the screen for as long as the address
 * bar is showing, and `html` is `overflow: hidden`, so the bottom of the app is simply cut off.
 */
test('the shell leaves a usable canvas at this geometry', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const vp = page.viewportSize()!;
	const pane = (await page.locator('.svelte-flow__pane').first().boundingBox())!;

	expect(
		outside(pane, { x: 0, y: 0, width: vp.width, height: vp.height }),
		'the canvas is inside the screen, not cut off below it'
	).toBeLessThanOrEqual(1);
	const floor = 2 * (await hit(page));
	expect(pane.height, 'and it is taller than a pair of tap targets').toBeGreaterThanOrEqual(floor);
	expect(pane.width, 'and wider').toBeGreaterThanOrEqual(floor);

	// Rendered is not the same as reachable: `emptySpot` only resolves where the pane really is the
	// topmost element, which is what a long press to add a node needs.
	await emptySpot(page);
});
