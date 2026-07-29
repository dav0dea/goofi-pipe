import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * The app header's progressive overflow (D-R6) and the canvas commands it carries (D-R4).
 *
 * This is a DESKTOP spec on purpose. The collapse keys on available WIDTH, not on device class —
 * a 900px desktop window has exactly the phone's problem — and the menu is resident chrome at
 * every width, because the commands inside it (delete, group, select-all, copy/paste/duplicate,
 * multi-select) have no bar slot to lose and are otherwise reachable only by a keyboard chord.
 *
 * The three traps D-R6 names are pinned as arithmetic in `editor/overflowFit.test.ts`; what this
 * file proves is that the real bar is wired to that arithmetic — that items really leave in the
 * declared order, that the result is stable when the width crosses a boundary and comes back, and
 * that nothing the header used to do changed.
 */

/** Priority order, HIGHEST first — the bar keeps these longest, so it gives them up in reverse. */
const PRIORITY = ['topbar-undo', 'topbar-redo', 'topbar-save', 'topbar-load', 'topbar-save-caret'];
const SPILL_ORDER = [...PRIORITY].reverse();

/** Which header actions are currently rendered in the bar (a spilled one is `display: none`). */
async function inBar(page: Page): Promise<string[]> {
	return page.locator('.topbar .actions').evaluate((el) =>
		[...el.querySelectorAll<HTMLElement>('button[data-testid]')]
			.filter((b) => b.offsetParent !== null)
			.map((b) => b.dataset.testid!)
	);
}

/** Resize and let the ResizeObserver settle (it runs after layout, before the next paint). */
async function widthTo(page: Page, width: number): Promise<void> {
	await page.setViewportSize({ width, height: 720 });
	await page.evaluate(
		() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
	);
}

async function openOverflow(page: Page): Promise<void> {
	await page.getByTestId('topbar-overflow').click();
	await expect(page.locator('.context-menu').first()).toBeVisible();
}

/** One menu row by its exact label. Not `getByRole('menuitem', { name })`: a row's accessible
 * name also carries its icon glyph and its ✓ marker, so the name is not the label. */
function menuRow(page: Page, label: string): Locator {
	return page
		.locator('.context-menu .item')
		.filter({ has: page.locator('.label', { hasText: new RegExp(`^${label}$`) }) });
}

test('the overflow trigger is resident chrome, outside the pinned action list', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const trigger = page.getByTestId('topbar-overflow');
	await expect(trigger).toBeVisible();
	// `topbar.spec.ts` pins `.topbar .actions button` as an exact ordered list of the five
	// app-global actions. The trigger is not one of them, so it must not live in that box.
	await expect(page.locator('.topbar .actions [data-testid="topbar-overflow"]')).toHaveCount(0);
});

test('a wide window keeps every action in the bar', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 1400);
	expect((await inBar(page)).sort()).toEqual([...PRIORITY].sort());
});

test('actions leave one at a time, lowest priority first', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	const left: string[] = [];
	let prev = new Set<string>(PRIORITY);
	for (let w = 1400; w >= 320; w -= 20) {
		await widthTo(page, w);
		const now = new Set(await inBar(page));
		for (const id of now) {
			expect(prev.has(id), `${id} came BACK into the bar at ${w}px`).toBe(true);
		}
		for (const id of prev) if (!now.has(id)) left.push(id);
		prev = now;
	}
	// One at a time, in the declared order — a prefix of it, since how many fit at 320px depends
	// on the root size the responsive clamp lands on, which is not this test's question.
	expect(left.length, 'the bar does give actions up as it narrows').toBeGreaterThanOrEqual(3);
	expect(left, 'and it gives them up in the declared priority order').toEqual(
		SPILL_ORDER.slice(0, left.length)
	);
});

/** The menu rows a spilled action turns into. The caret is not one action but the two options
 * behind it. */
const AS_ROWS: Record<string, string[]> = {
	'topbar-undo': ['Undo'],
	'topbar-redo': ['Redo'],
	'topbar-save': ['Save'],
	'topbar-save-caret': ['Save As…', 'Save in browser'],
	'topbar-load': ['Load…']
};

test('every spilled action is reachable in the overflow menu — and only those', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 320);
	const kept = await inBar(page);
	expect(kept.length, 'something spilled at 320px').toBeLessThan(PRIORITY.length);

	await openOverflow(page);
	for (const id of PRIORITY) {
		const spilledHere = !kept.includes(id);
		for (const label of AS_ROWS[id]) {
			// Present exactly when the bar gave it up: a row that duplicates a visible button is
			// two doors onto one action, which is how the two representations drift apart.
			await expect(menuRow(page, label), `${label} (${id} spilled: ${spilledHere})`).toHaveCount(
				spilledHere ? 1 : 0
			);
		}
	}
});

/* Trap 1, at the real bar rather than at the arithmetic: moving an item out changes the bar's
   content, which re-fires the observer that moved it. A plan that read the bar's own width would
   flip-flop forever at exactly the boundary. */
test('the bar is stable across a boundary width, in both directions', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);

	// Find the first width at which something spills, then straddle it.
	let boundary = 0;
	for (let w = 1400; w >= 320; w -= 4) {
		await widthTo(page, w);
		if ((await inBar(page)).length < PRIORITY.length) {
			boundary = w;
			break;
		}
	}
	expect(boundary, 'the bar does overflow somewhere between 320 and 1400').toBeGreaterThan(0);

	await widthTo(page, boundary);
	const at = (await inBar(page)).join();
	await widthTo(page, boundary + 24);
	const above = (await inBar(page)).join();
	for (let i = 0; i < 4; i++) {
		await widthTo(page, boundary);
		expect(await inBar(page), 'crossing back lands on the same answer').toEqual(at.split(','));
		await widthTo(page, boundary + 24);
		expect(await inBar(page)).toEqual(above.split(','));
	}

	// …and having settled, it stays settled with nobody touching it.
	await widthTo(page, boundary);
	await page.waitForTimeout(500);
	expect(await inBar(page)).toEqual(at.split(','));
});

test('the layout tab strip keeps a floor to spill against', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await widthTo(page, 412);
	// o1: the actions used to take every pixel and squeeze `.tabslot` to zero, so every layout tab
	// AND the ＋ that makes one were unreachable. Giving the actions a floor to spill below is
	// what fixes it.
	const slot = await page.locator('.topbar .tabslot').boundingBox();
	expect(slot!.width, 'the tab strip is not squeezed to nothing').toBeGreaterThan(40);
	await expect(page.getByRole('button', { name: 'New tab' })).toBeVisible();
});

test('the canvas commands live in the menu at every width', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	for (const w of [1400, 412]) {
		await widthTo(page, w);
		await openOverflow(page);
		for (const name of ['Select all', 'Delete selection', 'Group into sub-patch', 'Multi-select mode']) {
			await expect(menuRow(page, name), `${name} at ${w}px`).toBeVisible();
		}
		await page.keyboard.press('Escape');
	}
});

test('Select all from the menu selects the editor’s nodes', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await page.evaluate(() =>
		(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
	);
	await page.waitForFunction(
		(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
		uid
	);
	try {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await openOverflow(page);
		await menuRow(page, 'Select all').click();
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
			.toBeGreaterThan(0);
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
	}
});

test('multi-select mode is a mode: it stays on, and the header says so', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const trigger = page.getByTestId('topbar-overflow');
	await expect(trigger).toHaveAttribute('aria-pressed', 'false');

	await openOverflow(page);
	await menuRow(page, 'Multi-select mode').click();
	await expect(trigger, 'the always-visible chrome carries the mode').toHaveAttribute(
		'aria-pressed',
		'true'
	);

	// …and the row itself reads back as checked next time the menu is opened.
	await openOverflow(page);
	const row = menuRow(page, 'Multi-select mode');
	await expect(row.locator('.check')).toHaveText('✓');
	await row.click();
	await expect(trigger).toHaveAttribute('aria-pressed', 'false');
});
