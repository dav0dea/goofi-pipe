import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp } from '../lib/app';
import { outside, settledBox } from '../lib/geometry';
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
 * The façade's `save(path)` would reach the same state — the store is what publishes the new path
 * now, since the `save` arm broadcasts no `save_path_changed` (only `load` does) — but this file's
 * subject is what the header does at a real geometry, so it takes the door a user takes. (It used
 * to be the ONLY door: the `savePath` write lived in `AppShell`, above the seam the façade calls.)
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
	// A LIVE patch, not an idle one. `PerfHud` is `{#if active}` and shows nothing while no frames
	// flow, so a header guard written against the boot state measures a brand cluster that is
	// missing its widest member — the same trap as measuring it unnamed.
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	try {
		await waitForNode(page, uid);
		// Attached, not visible: `{#if active}` is what says frames are flowing, and below 520px the
		// bar then stands the HUD DOWN — which is the point. Asserting visibility here would be
		// asserting the defect.
		await expect(page.getByTestId('perf-hud'), 'frames are flowing, so the HUD is live').toBeAttached();
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

	// The filename is where the brand's shrink is absorbed — R made `.path` the sole absorber
	// because it is "the only one an ellipsis still leaves readable". But the two chips beside it
	// are `white-space: nowrap` with the default `min-width: auto`, so neither could give an inch,
	// and at 412px — the width R was built for — the ellipsis left `.path` 68px of a 60-character
	// name, five or six characters. `toHaveText` is green on that: it reads DOM text, not pixels.
	// Eight rem is about a dozen monospace characters — enough to tell two patches apart.
	const rem = await page.evaluate(() => parseFloat(getComputedStyle(document.documentElement).fontSize));
	const path = (await page.locator('.topbar .path').boundingBox())!;
	expect(path.width, 'the patch name is still readable').toBeGreaterThanOrEqual(8 * rem);

	// The tab strip is what the actions spill in favour of; squeezed to zero, no layout tab and not
	// even the ＋ that makes one can be reached (o1). TopBar reserves `2 × --hit` for it in the
	// overflow BUDGET, and the layout that follows does not honour that number: the brand competes
	// for the same slack, so with a long filename the strip lands at 54.4px at 412px (88.1 at 863,
	// 83.9 at 712). R's audit settled that as CORRECT rather than as a miss — the reservation is
	// what decides when an ACTION gives up its slot, and pinning `.tabslot` to the same number would
	// take those pixels back out of the filename, i.e. two mechanisms fighting over one budget. What
	// the strip must never go below is ONE tap target, which is what is asserted here; the strip is
	// its own `overflow-x` scroller with --hit-floored pills, so beyond that what does not fit is
	// scrolled to rather than lost.
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

/* Moved here from `touch-narrow.spec.ts` (which keeps the fixed-cost rows it was written for): the
   pane is clamped against its HOST, so what it leaves behind is a fraction of the editor panel — a
   different answer in portrait, in landscape and on a tablet.

   And since D-I2, so is WHICH EDGE it comes from: the anchor follows the host panel's own shape, so
   the three projects pointed at this file no longer all get the same one. `touch` (412×839) and
   `tablet` (712×1138) host a portrait panel and get the bottom sheet; `touch-landscape` (863×360)
   keeps the right-hand pane. The invariant under test is the same in both — the pane is flush with
   the edge it slid from, and leaves a full tap target of LIVE canvas on the other side. */
test('the inspector leaves the canvas it overlays reachable', async ({ page }) => {
	// The pane slides in over `--dur-slow`, and every number below is a POSITION — measured
	// mid-slide they describe a frame of the animation, not the layout the clamp produced. The app's
	// own reduced-motion rule collapses the transition, so this reads the settled pane.
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	try {
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		const pane = page.getByTestId('auto-side-panel');
		// `.open`, not merely visible: the pane is MOUNTED at every moment (`{#if enabled}`) and
		// parked at `translateX(100%)` until a selection lands, so `toBeVisible` resolves on a pane
		// still sitting off the right edge — and every position measured below would be that parked
		// one rather than the layout's. Then read it SETTLED: the class landing and the transform
		// having been applied are two different frames.
		await expect(pane).toHaveClass(/open/);
		const p = await settledBox(pane);
		const host = (await page.locator('.editor-panel').first().boundingBox())!;
		const h = await hit(page);
		// Asked of the HOST, exactly as `@container (orientation: portrait)` asks it — not of the
		// viewport, and not of the device class.
		const portrait = host.height >= host.width;

		// Before R its clamp was [260, 720] and never the host, so on a ~386px editor it covered the
		// canvas completely with its own far edge clipped off. Deselecting by tapping the canvas is
		// the only way to close it, so covering the canvas is a dead end.
		if (portrait) {
			expect(p.height, 'the sheet leaves canvas above it').toBeLessThan(host.height);
			expect(
				host.y + host.height - (p.y + p.height),
				'and is not clipped at its own edge'
			).toBeLessThan(2);
			expect(p.height, 'and never takes more than the 60% D-I6 allows').toBeLessThanOrEqual(
				host.height * 0.6 + 1
			);
			expect(p.y - host.y, 'the strip is at least one tap target tall').toBeGreaterThanOrEqual(h);
		} else {
			expect(p.width, 'the pane leaves a strip of canvas').toBeLessThan(host.width);
			expect(
				host.x + host.width - (p.x + p.width),
				'and is not clipped at its own edge'
			).toBeLessThan(2);
			expect(p.x - host.x, 'the strip is at least one tap target wide').toBeGreaterThanOrEqual(h);
		}

		// …and the strip has to be LIVE, not merely empty. The coarse resize band is a
		// `touch-action: none`, pointer-capturing overlay laid over the handle, so it reaches back
		// over the very canvas this clamp keeps free.
		//
		// LANDSCAPE: aimed half a --hit off the pane's left edge, and at whichever height there is
		// bare canvas, since a node can float anywhere in a strip this narrow. That aim point is the
		// strip's natural one, and it is exactly what a centred band killed — with the band centred
		// there was no such height, because it owned that whole column. The band leans INWARD here,
		// so the aim point is clear of it by construction.
		//
		// PORTRAIT: the band cannot lean inward (the sheet's top row is its ✕), so it leans fully
		// out and half a --hit off the edge is INSIDE it. The strip there is 40% of the host rather
		// than one tap target, so its natural aim point is not its rim — `emptySpot` finds a live
		// point in it, which is by construction neither under the sheet nor under the band, and
		// clear of the node card by a tap target so touch adjustment cannot snap the tap onto it.
		let spot: { x: number; y: number };
		if (portrait) spot = await emptySpot(page);
		else {
			const found = await page.evaluate(
				([px, top, bottom]) => {
					for (let cy = top + 20; cy < bottom - 20; cy += 16)
						if (document.elementFromPoint(px, cy)?.classList.contains('svelte-flow__pane'))
							return { x: px, y: cy };
					return null;
				},
				[p.x - h / 2, p.y, p.y + p.height]
			);
			expect(found, 'the strip’s aim point is canvas, not a resize band').not.toBeNull();
			spot = found!;
		}

		// …and the strip has to be a full tap target DEEP, which that probe cannot see: it aims
		// points, not an area. `.resize-handle` is absolutely positioned against `.side-panel`'s
		// PADDING box, and its coarse `::before` reaches further out still. The band pointer-captures
		// and `preventDefault`s, so every pixel of it is canvas the escape hatch does not really
		// have. Measured before the tap below, which deselects and parks the pane.
		const band = await page.getByTestId('panel-resize-handle').evaluate((el) => {
			const cs = getComputedStyle(el, '::before');
			if (cs.content === 'none') return null;
			const r = el.getBoundingClientRect();
			return { left: r.left + parseFloat(cs.left), top: r.top + parseFloat(cs.top) };
		});
		expect(band, 'the coarse grip band is there to measure').not.toBeNull();
		expect(
			portrait ? band!.top - host.y : band!.left - host.x,
			'the LIVE strip is a full tap target once the grip band is counted'
		).toBeGreaterThanOrEqual(h);

		await page.touchscreen.tap(spot.x, spot.y);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length), {
				message: 'a tap in the reserved strip reaches the canvas and deselects'
			})
			.toBe(0);
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
