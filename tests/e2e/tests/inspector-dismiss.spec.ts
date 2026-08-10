import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

/**
 * The selection inspector had no way out (D-R9, carryover **C9**).
 *
 * It slides in over its editor whenever exactly one node is selected. Its only off-switch — the
 * editor's `inspector-toggle` corner control — sits at `z-index: 5` UNDER the pane's 50, so the
 * moment the pane opens the control that closes it is buried; and the other exit, tapping empty
 * canvas to deselect, needs canvas the pane covers. On a phone that was a dead end. On desktop it
 * was merely obscure, which is the same defect (D-R2: one system, one fix).
 *
 * A `default`-project spec on purpose: desktop is the reference, and the fix is not coarse-gated.
 */

const pane = (page: Page) => page.getByTestId('auto-side-panel');

async function expectParked(page: Page, message = 'the inspector is parked'): Promise<void> {
	await expect(pane(page), message).not.toHaveClass(/open/);
	await expect(pane(page), `${message} and is no longer painted`).toHaveCSS('visibility', 'hidden');
}

async function addAndSelect(page: Page): Promise<string> {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(pane(page), 'a single selection opens the inspector').toHaveClass(/open/);
	return uid;
}

test.afterEach(async ({ page }) => {
	await page.evaluate(() => {
		const g = (window as any).goofi;
		const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
		if (uids.length) return g.commands.removeNodes(uids);
	});
});

test('the inspector has a dismiss control, and it closes the pane', async ({ page }) => {
	await addAndSelect(page);
	const panel = pane(page);
	const close = panel.getByTestId('inspector-close');
	await expect(close, 'the pane carries its own way out').toBeVisible();
	await expect(panel).toHaveCSS('transform', 'matrix(1, 0, 0, 1, 0, 0)');
	const openingDuration = await panel.evaluate((el) => getComputedStyle(el).transitionDuration.split(',')[0]);
	await close.click();
	await expect(panel, 'closing starts the outgoing state').not.toHaveClass(/open/);
	const outgoing = await panel.evaluate((el) => {
		const cs = getComputedStyle(el);
		return { visibility: cs.visibility, duration: cs.transitionDuration.split(',')[0] };
	});
	expect(outgoing.visibility, 'the pane remains painted while it slides out').toBe('visible');
	expect(outgoing.duration, 'open and close use the same motion duration').toBe(openingDuration);
	await expectParked(page, 'the completed outro hands the canvas back');
});

test('the ✕ is a close, not an off-switch: the next selection brings the pane back', async ({
	page
}) => {
	const uid = await addAndSelect(page);
	await pane(page).getByTestId('inspector-close').click();
	await expectParked(page, 'dismissed');

	// Deselect, re-select the SAME node → the pane returns. A dismissal is scoped to the
	// selection it was made in, not to the editor's lifetime.
	await page.evaluate(() => (window as any).goofi.commands.select([]));
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(pane(page), 're-selecting revives the pane').toHaveClass(/open/);

	// Dismiss again, then select a DIFFERENT node directly → the pane returns for it.
	await pane(page).getByTestId('inspector-close').click();
	await expectParked(page);
	const other = await addNode(page, 'Buffer', 'inputs');
	await waitForNode(page, other);
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), other);
	await expect(pane(page), 'a different node revives the pane').toHaveClass(/open/);

	// The ◧, by contrast, IS the off-switch: turn the inspector off with it and selection
	// changes stay silent.
	await pane(page).getByTestId('inspector-close').click();
	await expectParked(page);
	await page.getByTestId('inspector-toggle').click(); // shows (clears the dismissal)
	await expect(pane(page)).toHaveClass(/open/);
	await page.evaluate(() => (window as any).goofi.commands.select([]));
	await page.getByTestId('inspector-toggle').click(); // parked + visible → this press disables
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expectParked(page, 'disabled stays disabled across selections');
});

test('the dismiss control lives IN the identity header, right-most, after the state badge', async ({
	page
}) => {
	await addAndSelect(page);
	// Let the slide-in SETTLE before reading geometry: two sequential boundingBox reads during the
	// transition see the pane at two different positions, which once inverted this very assertion.
	await expect(pane(page)).toHaveCSS('transform', 'matrix(1, 0, 0, 1, 0, 0)');
	// The ✕ shares the identity Bar with the node's state badge — it does not get a strip of its
	// own above the header (that strip spent a full row saying nothing else). It sits at the far
	// corner, AFTER the badge, where every panel's ✕ already lives.
	const bar = pane(page).locator('.ui-bar', { has: page.getByTestId('node-state') });
	const close = bar.getByTestId('inspector-close');
	await expect(close, 'the ✕ is a resident of the identity Bar').toBeVisible();
	const closeBox = (await close.boundingBox())!;
	const badgeBox = (await bar.getByTestId('node-state').boundingBox())!;
	expect(
		badgeBox.x + badgeBox.width,
		'the running/error badge sits left of the ✕'
	).toBeLessThanOrEqual(closeBox.x + 1);
	const barBox = (await bar.boundingBox())!;
	expect(
		barBox.x + barBox.width - (closeBox.x + closeBox.width),
		'…and the ✕ is the bar’s right-most element'
	).toBeLessThanOrEqual(16);
	const closeCenter = closeBox.y + closeBox.height / 2;
	const badgeCenter = badgeBox.y + badgeBox.height / 2;
	expect(Math.abs(closeCenter - badgeCenter), 'on the same row').toBeLessThanOrEqual(2);
});

test('the buried toggle is not left under the pane, and it brings the pane back', async ({
	page
}) => {
	const uid = await addAndSelect(page);
	// While the pane is open the toggle is covered at every width (the pane is pinned to the right
	// edge and the toggle sits 10px inside it), so leaving it in the tree only means an invisible,
	// tabbable control under an opaque surface.
	await expect(page.getByTestId('inspector-toggle')).toHaveCount(0);

	await pane(page).getByTestId('inspector-close').click();
	const toggle = page.getByTestId('inspector-toggle');
	await expect(toggle, 'closing hands the affordance back').toBeVisible();
	await expect(toggle).toHaveAttribute('aria-pressed', 'false');

	// …and it is a toggle, not a one-way door: pressing it re-arms the pane for the same selection.
	await toggle.click();
	await expect(pane(page), 'the same node is inspected again').toHaveClass(/open/);
	await expect(pane(page).getByTestId('node-name')).toHaveText(
		(await page.evaluate((u) => (window as any).goofi.query.node(u).name, uid)) as string
	);
});

test('Escape closes the inspector by clearing the selection', async ({ page }) => {
	await addAndSelect(page);
	// The editor's own keydown handler answers Escape when focus is not in a field; clearing the
	// selection is what closes the pane. Pinned because it is the keyboard exit D-R9 says is
	// missing — it is not, and a future guard must not quietly take it away.
	await page.locator('.svelte-flow__pane').first().focus().catch(() => {});
	await page.keyboard.press('Escape');
	await expect(pane(page)).not.toHaveClass(/open/);
});

/**
 * The pane stays mounted so it can animate both directions; with nothing selected it sits hidden
 * and parked at `translateX(100%)`. The ✕ used to ride the pane itself (a
 * strip of chrome above the header), which left a focusable control inside the parked subtree for
 * an AT virtual cursor and any split-layout Tab order to reach. It now lives in ParamForm's
 * identity Bar, which only exists WITH a node — so the parked subtree holds no authored focusable
 * control at all, by construction rather than by a pointer-events guard.
 */
test('the parked inspector is neither painted nor reachable across an orientation change', async ({
	page
}) => {
	await page.goto('/');
	await waitForApp(page);
	// The pane is mounted (the inspector is enabled by default) but has nothing to inspect.
	await expect(pane(page), 'parked, not unmounted').toHaveCount(1);
	await expect(pane(page)).not.toHaveClass(/open/);
	await expect(pane(page), 'the parked box is not painted').toHaveCSS('visibility', 'hidden');

	// Crossing the same portrait/landscape seam a phone rotation crosses changes the parked
	// transform from Y to X. It must remain unpainted while that transform transition runs.
	await page.setViewportSize({ width: 412, height: 915 });
	await expect
		.poll(() =>
			pane(page).evaluate((el) =>
				getComputedStyle(el).getPropertyValue('--pane-axis').trim()
			)
		)
		.toBe('y');
	await page.setViewportSize({ width: 915, height: 412 });
	await expect
		.poll(() =>
			pane(page).evaluate((el) =>
				getComputedStyle(el).getPropertyValue('--pane-axis').trim()
			)
		)
		.toBe('x');
	await expect(pane(page), 'rotation does not reveal the parked transition').toHaveCSS(
		'visibility',
		'hidden'
	);

	await expect(
		pane(page).getByTestId('inspector-close'),
		'the parked pane carries no ✕ at all — the identity Bar only exists with a node'
	).toHaveCount(0);
	expect(
		await pane(page).evaluate(
			(el) => el.querySelectorAll('button, [tabindex]:not([tabindex="-1"])').length
		),
		'…and no other authored focusable control either'
	).toBe(0);
});
