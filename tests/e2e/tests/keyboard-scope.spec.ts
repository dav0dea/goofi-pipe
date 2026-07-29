import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';

/**
 * Who owns a key press — the editor panel, the browser, or the modal on top of it.
 *
 * `NodeEditorPanel`'s window-level `onKeydown` guards on two things only: is my panel the active
 * one, and is the target a text field. Neither question is the one that matters at these two
 * boundaries, and both were carried over from the M audit (R2-3 and R2-2) without ever entering a
 * task brief:
 *
 *  - **Tab** was `preventDefault`ed for EVERY non-field target, with no `shiftKey` check. The first
 *    Tab looks fine because it opens the add-node menu and the menu focuses its search — but
 *    `{#if menuOpen}` is unkeyed, so re-entering `openAddMenu` with the menu already open neither
 *    remounts it nor re-fires that focus. Every later Tab is a bare `preventDefault` with nothing
 *    to refocus: a one-way trap (WCAG 2.1.2), and the reason no chrome outside the canvas — the tab
 *    strip, the header actions, the inspector's ✕, R's own `.conn-label` focus reveal — was ever
 *    Tab-reachable.
 *  - The whole handler ignored the fact that a modal `<dialog>` was up. It opens with focus on its
 *    path field, which the allowlist does cover — but the first navigation click puts focus on a
 *    `<button>`, and load mode REQUIRES one (`fs-open` is `disabled={!selected}`). From there
 *    Escape dismissed the dialog AND ran the editor's own Escape ladder behind it, and Ctrl+A
 *    selected every node on the hidden canvas.
 *
 * The standdown is read from the DOM (`closest('dialog[open]')`), NOT from `ui().modalOpen`: that
 * flag is a ref-count an in-panel fx editor raises too, and standing the canvas down for a merely
 * expanded inspector field would be a different bug. The last test here is that trade, pinned.
 */

/** A short, stable name for whatever currently has focus. */
const activeName = (page: Page): Promise<string | null> =>
	page.evaluate(() => {
		const el = document.activeElement as HTMLElement | null;
		if (!el) return null;
		return el.dataset.testid ?? el.tagName.toLowerCase();
	});

const selectedNodes = (page: Page): Promise<string[]> =>
	page.evaluate(() => (window as any).goofi.query.selection().nodes);

/**
 * Open the Load browser and leave focus on a real control inside it.
 *
 * `showModal()` lands on the path field, which the handler's INPUT allowlist happens to cover — so
 * the reachable state is one navigation click later. Clicking a root is the first thing load mode
 * asks of anyone, and every route through it (a root, a directory row, a `.gfi` row) ends with
 * focus on a `<button>`.
 */
async function openBrowserOnAButton(page: Page): Promise<ReturnType<Page['getByTestId']>> {
	await page.getByTestId('topbar-load').click();
	const modal = page.getByTestId('fs-browser');
	await expect(modal).toBeVisible();
	await modal.locator('nav.roots button.root').first().click();
	expect(await activeName(page), 'focus is on one of the dialog’s buttons, not its path field').toBe(
		'button'
	);
	return modal;
}

test('Tab on a chrome control is the browser’s, not the canvas’s', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// The node editor is the active panel from the first frame (DEFAULT_PANEL_TYPE + `_focusFirst`),
	// and no keyboard event ever clears that — which is why this reproduces in the boot layout.
	await page.getByTestId('topbar-load').focus();
	await page.keyboard.press('Tab');

	await expect(
		page.getByTestId('add-node-menu-anchor'),
		'the canvas does not claim a Tab it was not given'
	).toHaveCount(0);
	expect(await activeName(page), 'focus moved on to the next control in the header').toBe(
		'topbar-overflow'
	);
});

test('focus can leave the add-node menu again', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
	const menu = page.getByTestId('add-node-menu-anchor');
	await expect(menu).toBeVisible();
	await expect(page.getByTestId('add-menu-search'), 'the menu takes focus on open').toBeFocused();

	const outside = (): Promise<boolean> =>
		page.evaluate(() => {
			const a = document.activeElement;
			const m = document.querySelector('[data-testid="add-node-menu-anchor"]');
			return !!a && !!m && !m.contains(a);
		});
	let left = false;
	// Bounded: the menu holds a search field plus one row per palette entry, so a few dozen presses
	// is generous. Without the fix this never leaves — the second Tab lands on a row <button> and
	// every press after it is a bare preventDefault.
	for (let i = 0; i < 60 && !left; i++) {
		await page.keyboard.press('Tab');
		left = await outside();
	}
	expect(left, 'Tab is not a one-way door into the palette (WCAG 2.1.2)').toBe(true);

	await page.keyboard.press('Escape');
	await expect(menu).toHaveCount(0);
});

test('Tab on the bare canvas still opens the add-node menu', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	// The Blender-style shortcut is the point of the branch and must survive its scoping: with
	// nothing focused, the canvas is what the key press is for.
	await page.keyboard.press('Tab');
	const menu = page.getByTestId('add-node-menu-anchor');
	await expect(menu).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(menu).toHaveCount(0);
});

test('Escape dismisses the file browser without also running the editor’s ladder', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	try {
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		expect(await selectedNodes(page)).toEqual([uid]);

		const modal = await openBrowserOnAButton(page);

		await page.keyboard.press('Escape');
		await expect(modal, 'the reflex dismissal closes the dialog').toBeHidden();
		expect(
			await selectedNodes(page),
			'…and only the dialog — the canvas behind it kept its selection'
		).toEqual([uid]);
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});

test('Ctrl+A does not reach the canvas behind the file browser', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, uid);
	try {
		await openBrowserOnAButton(page);
		await page.keyboard.press('Control+a');
		// A non-event needs a settle window; the selection write is synchronous, so this is orders of
		// magnitude more than it would need.
		await page.waitForTimeout(300);
		expect(await selectedNodes(page), 'the hidden canvas selected nothing').toEqual([]);
	} finally {
		await page.keyboard.press('Escape');
		await expect(page.getByTestId('fs-browser')).toBeHidden();
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}
});

/**
 * The trade the modal guard must NOT overshoot. `ui().modalOpen` is a ref-count that
 * `inspector/ParamField.svelte` raises for a merely-expanded fx editor, so the naive
 * `if (ui().modalOpen) return;` would stand the canvas down while an inspector field is open —
 * which is why R2-2 was deferred to R in the first place. A dialog is a dialog; an fx editor is a
 * panel that happens to hold a textarea.
 */
test('an expanded fx editor holds the global standdown but not the editor’s own keys', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);
	const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
	await waitForNode(page, buf);
	try {
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), osc);
		const pane = page.getByTestId('auto-side-panel');
		await expect(pane).toHaveClass(/open/);
		await pane.getByTestId('param-fx-toggle').first().click();
		await pane.getByTestId('param-expr-expand').first().click();
		await expect(pane.getByTestId('param-expr-multiline')).toBeVisible();
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.modalOpen()))
			.toBe(true);

		// Focused, not clicked: a click on the collapse chip would close the very editor under test.
		await pane.getByTestId('param-expr-collapse').first().focus();
		await page.keyboard.press('Control+a');
		await expect
			.poll(() => selectedNodes(page).then((n) => n.slice().sort()), {
				message: 'the editor’s own Select all still fires with an fx editor expanded'
			})
			.toEqual([osc, buf].sort());
	} finally {
		await page.evaluate(() => (window as any).goofi.commands.clearSelection());
		await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [osc, buf]);
		await waitForNoNode(page, osc).catch(() => {});
		await waitForNoNode(page, buf).catch(() => {});
	}
});
