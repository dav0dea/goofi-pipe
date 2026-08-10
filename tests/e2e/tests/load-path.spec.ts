import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import { addNode, nodes, waitForNode } from '../lib/goofi';

/**
 * Path-load only, and the `New` door — sub-project W's frontend half.
 *
 * A `.gfi` is a zip archive now, so a backend PATH is the only way in. The manager still answers
 * `load_text` (hand it a YAML string), but nothing in the browser can reach it: a zip through
 * `File.text()` is mojibake, so the upload button that was its one caller is gone. And the save
 * path is MANAGER-owned — the client no longer latches it off the `save` reply, so what this file
 * proves about `savePath` is that it arrives over the wire, not that a local write happened to run.
 */

let scratch = '';
test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-load-')));
});
test.afterAll(() => {
	fs.rmSync(scratch, { recursive: true, force: true });
});

test('a .gfi round-trips through the manager-owned save path', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	try {
		const target = path.join(scratch, 'roundtrip.gfi');
		const uid = await addNode(page, 'Oscillator');
		await waitForNode(page, uid);

		await page.evaluate((p) => (window as any).goofi.commands.save(p), target);
		// The store learned the path from the MANAGER, not from a client-side write.
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.toBe(target);
		expect(fs.existsSync(target), 'the .gfi landed on disk').toBe(true);

		// It is a zip archive now, not YAML.
		expect(fs.readFileSync(target).subarray(0, 2).toString('binary')).toBe('PK');

		await addNode(page, 'Buffer', 'signal'); // diverge
		await page.evaluate((p) => (window as any).goofi.commands.load(p), target);
		await expect.poll(async () => (await nodes(page)).map((n) => n.type)).toEqual(['Oscillator']);
		// A load names the patch too — same manager-owned path, arriving on the load's snapshot.
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.toBe(target);
	} finally {
		await resetPatch(page);
	}
});

test('New hands back an empty, unnamed, clean patch', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	try {
		const uid = await addNode(page, 'Oscillator');
		await waitForNode(page, uid);
		await page.evaluate((p) => (window as any).goofi.commands.save(p), path.join(scratch, 'named.gfi'));
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.not.toBe(null);

		await page.evaluate(() => (window as any).goofi.commands.newPatch());
		await expect.poll(async () => (await nodes(page)).length).toBe(0);
		// All three halves, because a New that forgot any one of them would leave the shared backend
		// in exactly the state this suite's other specs cannot start from.
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.toBe(null);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges))
			.toBe(false);
		expect(
			await page.evaluate(() => (window as any).goofi.query.canUndo()),
			'a New retires the undo stack — there is nothing to undo across it'
		).toBe(false);
	} finally {
		await resetPatch(page);
	}
});

/**
 * The failed-save toast, driven through the ONLY door that can raise it: the header's Save on an
 * already-named patch, which overwrites silently with no dialog in front of it. The façade's
 * `save()` rejects past `AppShell.triggerSave`'s catch, so this is a UI test or it is nothing —
 * and it is the whole reason the surface exists (a save onto a path since deleted, moved or made
 * read-only used to be a `console.error`).
 */
test('a save that fails says so, instead of failing in silence', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	const doomed = path.join(scratch, 'doomed');
	try {
		fs.mkdirSync(doomed);
		await page.evaluate(
			(p) => (window as any).goofi.commands.save(p),
			path.join(doomed, 'gone.gfi')
		);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
			.not.toBe(null);

		// Take the directory out from under the remembered path, then Save again.
		fs.rmSync(doomed, { recursive: true, force: true });
		await page.getByTestId('topbar-save').click();

		await expect(page.getByTestId('toast'), 'the rejection reached the alarm surface').toContainText(
			/Save failed/
		);
	} finally {
		fs.rmSync(doomed, { recursive: true, force: true });
		await resetPatch(page);
	}
});

/** The content door is gone from the CLIENT. Asserting the façade no longer offers it is what keeps
 *  a well-meaning re-add from silently re-introducing a call that can only ever ship mojibake. */
test('there is no content-load door left on the façade', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	expect(
		await page.evaluate(() => 'loadText' in (window as any).goofi.commands),
		'`loadText` went with the upload button; a .gfi is an archive, so a path is the only door'
	).toBe(false);
});
