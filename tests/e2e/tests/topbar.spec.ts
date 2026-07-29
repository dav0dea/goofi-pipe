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
 * Pinned as an exact ordered list rather than two absence assertions, because the order is also the
 * priority order D-R6's progressive overflow will spill from (lowest priority spills first).
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
