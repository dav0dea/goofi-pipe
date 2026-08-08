import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';

/**
 * What the app header is allowed to carry.
 *
 * The header is the only constant chrome, so it holds app-global actions: undo/redo of the whole
 * session, and the patch's save/load. It used to also hold ＋ Add node and Fit, which are neither —
 * both act on ONE node-editor panel (they resolve "the active editor" behind the user's back, and
 * silently pick an arbitrary one when several are open), and every editor panel already offers
 * both locally: the flow controls' own fit button, and four add-node doors of its own.
 *
 * Pinned as an exact ordered list rather than two absence assertions, so the header's contents and
 * their reading order are both fixed. This is the DOM order, which is NOT the priority order
 * D-R6's progressive overflow spills from — that is Undo · Redo · Save · Load… · Save▾ (the caret
 * leaves first, and the split degrades into a plain Save button), and `topbar-overflow` — the
 * resident menu the spilled actions land in — deliberately lives outside `.actions`.
 * `topbar-overflow.spec.ts` owns the spill order.
 */
const HEADER_ACTIONS = [
	'topbar-undo',
	'topbar-redo',
	'topbar-save',
	'topbar-save-caret',
	'topbar-load'
];

test('the app header carries exactly the app-global actions', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const ids = await page
		.locator('.topbar .actions button')
		.evaluateAll((els) => els.map((el) => el.getAttribute('data-testid')));
	expect(ids, 'panel-local behaviour does not belong in the app header').toEqual(HEADER_ACTIONS);
});

test('the save menu offers Save As and nothing else', async ({ page }) => {
	// "Save in browser" is gone (user decision, 2026-08-08): a save writes a backend file, full
	// stop. The caret stays a menu so the split control keeps its shape and its spill behaviour.
	await page.goto('/');
	await waitForApp(page);
	await page.getByTestId('topbar-save-caret').click();
	const rows = page.locator('.context-menu [role="menuitem"]');
	await expect(rows, 'one row: Save As…').toHaveCount(1);
	await expect(rows.first()).toHaveText(/Save As/);
	await expect(
		page.locator('.context-menu').getByText(/browser/i),
		'no browser-save row anywhere in it'
	).toHaveCount(0);
});

/**
 * …and what it is not allowed to carry: a brand.
 *
 * Phil's call, and both halves of it are the same point. The ⟁ is not goofi's logo — it never was —
 * and the wordmark spends ~95px of a 412px bar restating what the browser tab already says, in the
 * one strip of chrome that is on screen at every width. The bar is for what the user can DO here.
 *
 * Read off `textContent`, not `innerText`: the wordmark was hidden below 520px by a container
 * query, so a visible-text assertion would already have been green on a phone while the brand was
 * still in the DOM taking part in the layout.
 */
test('the app header carries no brand', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const text = (await page.locator('.topbar').textContent()) ?? '';
	expect(text, 'the wordmark is the browser tab’s job').not.toContain('goofi-pipe');
	expect(text, 'and the ⟁ was never the logo').not.toContain('⟁');
});
