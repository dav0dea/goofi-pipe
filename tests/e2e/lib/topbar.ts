import { expect, type Locator, type Page } from '@playwright/test';

/**
 * The app header's action inventory and the handles for reading it.
 *
 * Shared through `lib/` rather than by importing one spec from another, since loading a spec file
 * is how Playwright registers its tests. Three specs ask the same question of this bar at
 * different widths — `topbar-overflow.spec.ts` by resizing a desktop window, `touch-reflow.spec.ts`
 * at three real device geometries, `touch-authoring.spec.ts` while saving a patch at 412px — and
 * the priority order and the row labels have to be ONE list, or two of them can green while the
 * bar has quietly grown a sixth action nobody checks.
 */

/** Priority order, HIGHEST first — the bar keeps these longest, so it gives them up in reverse. */
export const PRIORITY = [
	'topbar-undo',
	'topbar-redo',
	'topbar-save',
	'topbar-load',
	'topbar-save-caret'
];

/** The overflow-menu row(s) each bar action becomes once it spills. The Save caret is a submenu
 *  in the bar, so it spills as its leaf row rather than as itself. */
export const AS_ROWS: Record<string, string[]> = {
	'topbar-undo': ['Undo'],
	'topbar-redo': ['Redo'],
	'topbar-save': ['Save'],
	'topbar-save-caret': ['Save As…'],
	'topbar-load': ['Load…']
};

/** Which header actions are currently rendered in the bar (a spilled one is `display: none`). */
export function inBar(page: Page): Promise<string[]> {
	return page.locator('.topbar .actions').evaluate((el) =>
		[...el.querySelectorAll<HTMLElement>('button[data-testid]')]
			.filter((b) => b.offsetParent !== null)
			.map((b) => b.dataset.testid!)
	);
}

/**
 * The header's action list, once its ResizeObserver-driven re-plan has stopped moving.
 *
 * Naming the patch changes the status cluster's width, and so does resizing the window — either
 * re-fires the observer that decides what spills, and where the header's flex line is already
 * overflowing the cluster and the tab strip give way TOGETHER, so the plan can take several rounds
 * to converge. A single read can therefore catch it mid-flight and report an action as "in the bar"
 * a frame before it is `display: none`. Two agreeing reads is also the assertion that it settles at
 * all, which is trap 1 (oscillation) seen from the outside.
 */
export async function settledBar(page: Page): Promise<string[]> {
	let prev = (await inBar(page)).join();
	for (let i = 0; i < 20; i++) {
		await page.waitForTimeout(50);
		const now = (await inBar(page)).join();
		if (now === prev) return prev === '' ? [] : prev.split(',');
		prev = now;
	}
	throw new Error(`the header's overflow plan never settled (last: ${prev})`);
}

export async function openOverflow(page: Page): Promise<void> {
	await page.getByTestId('topbar-overflow').click();
	await expect(page.locator('.context-menu').first()).toBeVisible();
}

/** One menu row by its exact label. Not `getByRole('menuitem', { name })`: a checkable row is a
 *  `menuitemcheckbox`, so one role name cannot reach every row. (The accessible name IS the label
 *  now — the glyph spans are `aria-hidden` — and `topbar-overflow.spec.ts` asserts that.) */
export function menuRow(page: Page, label: string): Locator {
	return page
		.locator('.context-menu .item')
		.filter({ has: page.locator('.label', { hasText: new RegExp(`^${label}$`) }) });
}

/**
 * Open the file browser through **Save As**, which is the only door onto it once the patch has a
 * name — a plain Save then overwrites silently.
 *
 * Save As has TWO doors and which one exists depends on the width: it is a row of the Save
 * split-button's dropdown while the caret is in the bar, and a row of the ⋯ overflow menu after the
 * caret spills — which it does FIRST among the actions, so at 320/412px only the ⋯ route is left.
 * It is the same DOM in both, so `menuRow` reaches it either way; the wording is already this
 * suite's own, pinned by `AS_ROWS` and asserted for this very row in `topbar-overflow.spec.ts`, so
 * a rename fails there by name rather than here as a missing element.
 */
export async function openSaveAs(page: Page): Promise<void> {
	const caret = page.getByTestId('topbar-save-caret');
	if (await caret.isVisible()) await caret.click();
	else await openOverflow(page);
	await menuRow(page, 'Save As…').click();
}
