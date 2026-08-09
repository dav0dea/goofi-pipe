import { test, expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import { addNode, waitForNode } from '../lib/goofi';

/**
 * The dirty taxonomy (R spec §4 / D-R3): *navigation* must not mark the patch unsaved, *authoring*
 * must. Both halves ride the same debounced `set_layout` push — the whole point is that the write
 * is classified, not the device — so both are asserted here through the real UI on one platform,
 * and the frontend unit tests pin the classification itself.
 *
 * Timing matters: the push is debounced 400ms, so "still clean" is only meaningful AFTER that
 * window has elapsed and the RPC has landed. Each assertion therefore waits past the debounce
 * rather than reading immediately — and each test leaves the workspace as it found it, because the
 * suite shares one backend and a leaked layout breaks later specs.
 */

/** Comfortably past AppShell's 400ms layout debounce plus the RPC round trip. */
const PAST_DEBOUNCE = 1200;

let scratch = '';
const patchName = `dirty-taxonomy-${process.pid}-${Date.now()}.gfi`;

test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-dirty-')));
});
test.afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }));

function unsavedChanges(page: Page): Promise<boolean> {
	return page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges);
}

/** Save the patch to the scratch file and wait for the manager to report it clean. */
async function saveClean(page: Page): Promise<void> {
	await page.evaluate((p) => (window as any).goofi.commands.save(p), path.join(scratch, patchName));
	await expect.poll(() => unsavedChanges(page), { message: 'a save makes it clean' }).toBe(false);
}

function firstPanelId(page: Page): Promise<string> {
	return page.evaluate(() => (window as any).goofi.query.panels()[0].panelId);
}

test('entering and leaving a sub-patch never dirties the patch', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.

	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);
	const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
	await waitForNode(page, buf);
	const inst: string = await page.evaluate(
		([a, b]) => (window as any).goofi.commands.groupNodes([a, b], [120, 120]),
		[osc, buf] as const
	);
	const group = page.getByTestId('subpatch-node');
	await expect(group, 'the two nodes became one sub-patch facade').toBeVisible();

	await saveClean(page);

	// ENTER — the real door: a double-click on the group node.
	await group.dblclick();
	const crumbs = page.getByTestId('subpatch-breadcrumb');
	await expect(crumbs, 'the editor descended into the sub-patch').toBeVisible();
	await page.waitForTimeout(PAST_DEBOUNCE);
	expect(await unsavedChanges(page), 'entering a sub-patch is navigation, not an edit').toBe(false);

	// LEAVE — the breadcrumb back to the top level. Same axis, same answer.
	await crumbs.getByRole('button', { name: 'Patch', exact: true }).click();
	await expect(crumbs, 'the editor climbed back out').toBeHidden();
	await page.waitForTimeout(PAST_DEBOUNCE);
	expect(await unsavedChanges(page), 'leaving a sub-patch is navigation too').toBe(false);

	// Dissolve the facade before wiping: `removeNodes` takes the members with it but leaves the
	// empty instance behind, and a leaked sub-patch is a second `subpatch-node` for the next run.
	await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
	await expect(group, 'the sub-patch facade is gone').toHaveCount(0);
	// `resetPatch`, not `clearGraph`: `saveClean` above NAMED the patch, and the name is the
	// manager's now — it would otherwise ride into every later spec and turn its Save into a
	// silent overwrite. (This file is #11 of the run; that is exactly how it happened.)
	await resetPatch(page);
});

test('changing a docked viewer’s type DOES dirty the patch', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.

	const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
	await waitForNode(page, osc);

	// Borrow the single panel as a Viewer bound to the oscillator, then save — so the ONLY thing
	// left to change is the viewer type itself.
	const panelId = await firstPanelId(page);
	await page.evaluate(
		([pid, uid]) => {
			(window as any).goofi.commands.setPanelType(pid, 'viewer');
			(window as any).goofi.commands.bindNodeToPanel(pid, uid);
		},
		[panelId, osc] as const
	);
	const kind = page.locator('.viewer-controls select.kind');
	await expect(kind, 'the panel is showing the oscillator with its type dropdown').toBeVisible();
	// Let the SETUP's own push land before saving. Both calls above are authoring, and the folded
	// intent only resets when the debounced push takes it — so without this wait a save clears the
	// flag while an `authored` is still pending, and the push that lands ~167ms later dirties the
	// patch no matter how the viewer-kind write is classified. The assertion below would then be
	// green against a viewer kind reclassified as navigation, which is the one thing it exists to
	// catch.
	await page.waitForTimeout(PAST_DEBOUNCE);
	await saveClean(page);

	await kind.selectOption('image');
	await expect
		.poll(() => unsavedChanges(page), { message: 'picking a viewer type is authoring' })
		.toBe(true);

	// Hand the workspace back: the type swap discards the panel's viewer state with it.
	await page.evaluate(
		(pid) => (window as any).goofi.commands.setPanelType(pid, 'node-editor'),
		panelId
	);
	await page.waitForTimeout(PAST_DEBOUNCE);
	await resetPatch(page); // …and the NAME `saveClean` gave it.
});
