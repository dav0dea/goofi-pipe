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
 *  in the bar, so it spills as its two leaf rows rather than as one. */
export const AS_ROWS: Record<string, string[]> = {
	'topbar-undo': ['Undo'],
	'topbar-redo': ['Redo'],
	'topbar-save': ['Save'],
	'topbar-save-caret': ['Save As…', 'Save in browser'],
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
