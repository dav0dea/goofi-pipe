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

test('the app header follows the fine-pointer control height', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const bar = (await page.locator('.topbar').boundingBox())!;
	const control = (await page.getByTestId('topbar-overflow').boundingBox())!;
	expect(bar.height, 'the bar adds no fixed vertical slack around its tallest control').toBe(
		control.height
	);
	expect(bar.height, 'desktop chrome does not reserve the touch-sized height').toBeLessThan(44);
});

test('the app header carries exactly the app-global actions', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const ids = await page
		.locator('.topbar .actions button')
		.evaluateAll((els) => els.map((el) => el.getAttribute('data-testid')));
	expect(ids, 'panel-local behaviour does not belong in the app header').toEqual(HEADER_ACTIONS);
});

test('the header is one icon family on one chrome text size (Phil: de-clutter)', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);

	// Every bar action is a GLYPH with an accessible name — Save/Load joined undo/redo, so the
	// action cluster reads as one family instead of two idioms side by side. The words live on
	// in the tooltips and the overflow-menu rows.
	for (const id of ['topbar-undo', 'topbar-redo', 'topbar-save', 'topbar-load']) {
		const btn = page.getByTestId(id);
		expect(
			(await btn.textContent())?.trim(),
			`${id} carries a glyph, not a word`
		).toBe('');
		await expect(btn.locator('svg'), `${id} renders its icon`).toHaveCount(1);
		expect(await btn.getAttribute('aria-label'), `${id} keeps its name for AT`).toBeTruthy();
	}

	// The bar's remaining TEXT (the tab labels, the patch name, the fps) shares ONE chrome
	// size — integer-snapped, so every baseline in the bar lands on the same device row.
	// The fluid clamp's fractional sizes snapped per element and scattered the ink ±0.75px.
	const sizeOf = (sel: string): Promise<string> =>
		page.locator(sel).first().evaluate((el) => getComputedStyle(el).fontSize);
	const tab = await sizeOf('[data-testid="workspace-tabs"] .ui-tab-label');
	expect(parseFloat(tab) % 1, `the chrome size (${tab}) snaps to whole pixels`).toBe(0);

	// The name chip and the fps readout render once the patch is live — adding a node dirties
	// it (● untitled) and starts frames flowing, which is both preconditions at once.
	const uid = await page.evaluate(() =>
		(window as any).goofi.commands.addNode('Oscillator', 'inputs', [40, 40])
	);
	try {
		const path = page.getByTestId('topbar-path');
		await path.waitFor({ state: 'attached' });
		expect(
			await path.evaluate((el) => getComputedStyle(el).fontSize),
			'the patch name shares the chrome size'
		).toBe(tab);

		// The fps readout is quiet text like the name beside it — the boxed pill is gone.
		const hud = page.getByTestId('perf-hud');
		await hud.waitFor({ state: 'attached' });
		const skin = await hud.evaluate((el) => {
			const cs = getComputedStyle(el);
			return { bg: cs.backgroundColor, fs: cs.fontSize };
		});
		expect(
			/rgba\(0, 0, 0, 0\)|transparent/.test(skin.bg),
			`the fps readout (${skin.bg}) is unboxed`
		).toBe(true);
		expect(skin.fs, 'the fps readout shares the chrome size').toBe(tab);
		const drops = hud.getByTestId('perf-drops');
		await expect(drops, 'the drop-rate slot stays mounted instead of shifting FPS').toBeVisible();
		await expect(drops).toHaveText(/^\s*\d+\/s\s*$/);

		// Identity and action items share one overflow group and one visual rhythm. Measure the INK
		// rather than the outer boxes: the text items deliberately borrow the clear space that each
		// IconButton's square puts around its glyph.
		const rhythm = await page.locator('.topbar .actions').evaluate((group) => {
			const box = (selector: string): DOMRect =>
				group.querySelector(selector)!.getBoundingClientRect();
			const fps = box('[data-testid="perf-hud"]');
			const name = box('.path-value');
			const undo = box('[data-testid="topbar-undo"] svg');
			const redo = box('[data-testid="topbar-redo"] svg');
			return {
				members: [
					'topbar-hud',
					'topbar-path',
					'topbar-undo',
					'topbar-redo',
					'topbar-save',
					'topbar-save-caret',
					'topbar-load'
				].every((id) => group.querySelector(`[data-testid="${id}"]`)),
				gaps: [name.left - fps.right, undo.left - name.right, redo.left - undo.right]
			};
		});
		expect(rhythm.members, 'fps, name and icon actions live in one spillable group').toBe(true);
		expect(Math.min(...rhythm.gaps), 'no identity ink sits flush against its neighbour').toBeGreaterThan(
			12
		);
		expect(
			Math.max(...rhythm.gaps) - Math.min(...rhythm.gaps),
			`text/text, text/icon and icon/icon gaps share one rhythm (${rhythm.gaps.join(', ')})`
		).toBeLessThanOrEqual(2);

		// The HUD is not merely in the same flex box; it participates in the same relocation plan.
		await page.setViewportSize({ width: 320, height: 720 });
		await expect(page.getByTestId('topbar-hud'), 'the FPS item spills as the group narrows').toBeHidden();
		await page.getByTestId('topbar-overflow').click();
		const fpsRow = page.getByRole('menuitem', { name: /^\d+ fps$/ });
		await expect(fpsRow, 'the same FPS information moves into the overflow menu').toHaveCount(1);
		await expect(fpsRow, 'information relocates without becoming an action').toBeDisabled();
	} finally {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
	}
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
