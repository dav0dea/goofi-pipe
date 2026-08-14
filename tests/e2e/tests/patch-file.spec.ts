import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import { addNode, nodes, waitForNode } from '../lib/goofi';

/**
 * `/patch.gfi` — the patch carried by the BROWSER, in both directions.
 *
 * This is the door onto locations the backend cannot reach. Running in a container goofi sees only
 * what was bind-mounted, and no `docker run` flag means "the whole host filesystem" on Linux, macOS
 * and Windows alike; the browser runs ON the host and its own dialogs reach anywhere. So the file
 * list in the modal shows what the SERVER can see, and these two buttons go around it.
 *
 * It has to be an e2e. The transport half is unit-tested (`api/patchFile.test.ts`) and the routes
 * are covered against a live server (`goofi-bridge/tests/patch_file.rs`), but the parts that
 * actually break — a hidden `<input type=file>` that the button must reach, a download that must
 * arrive as an attachment rather than navigating the SPA away — are Svelte glue, and this project
 * runs vitest without a DOM.
 *
 * A REAL round trip, not "a dialog closed": build a graph, carry it out through the browser, wipe
 * the patch, carry it back in, and assert the graph returned. Nothing here touches the backend's
 * filesystem — which is the whole point, and is also what makes it hermetic.
 */

/** Where the downloaded `.gfi` lands. The BROWSER's disk conceptually; the same machine in
 *  practice, since `globalSetup` spawns the fleet locally. */
let scratch = '';

test.beforeAll(() => {
	scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-patchfile-')));
});

test.afterAll(() => {
	fs.rmSync(scratch, { recursive: true, force: true });
});

/**
 * Hand the patch back. One backend serves every spec on this worker, so a spec that ends holding
 * nodes leaves them for the next one — and `expectPristineWorkspace` then reds the INNOCENT file
 * that runs after it. This spec is a round trip, so it ends holding exactly the graph it just
 * re-imported; without this it leaks two nodes into everything downstream. (It did, once, on the
 * first full-suite run: 18 failures across nine unrelated files.)
 */
test.afterEach(async ({ page }) => {
	await resetPatch(page);
});

test('a patch carried out through the browser comes back through the browser', async ({ page }) => {
	await page.goto('/');
	await waitForApp(page);
	await resetPatch(page);

	const osc = await addNode(page, 'Oscillator');
	const buf = await addNode(page, 'Buffer');
	await waitForNode(page, osc);
	await waitForNode(page, buf);

	// --- out ---------------------------------------------------------------
	// A fresh patch has no remembered path, so the Save button opens the browser rather than
	// silently overwriting — the same door a user meets before they have ever named the patch.
	await page.getByTestId('topbar-save').click();
	const download = page.getByTestId('fs-download');
	await download.waitFor();

	const [file] = await Promise.all([page.waitForEvent('download'), download.click()]);
	// Named by the SERVER, through Content-Disposition — not guessed in the client.
	expect(file.suggestedFilename()).toMatch(/\.gfi$/);

	const saved = path.join(scratch, 'roundtrip.gfi');
	await file.saveAs(saved);
	const bytes = fs.readFileSync(saved);
	// A `.gfi` is a zip. Asserting the magic rather than just a non-empty file is what separates
	// "something downloaded" from "the patch downloaded" — an HTML error page is non-empty too.
	expect(bytes.subarray(0, 2).toString('latin1')).toBe('PK');

	// The page must still BE the app: an attachment download must not navigate the SPA away.
	await expect(page.getByTestId('fs-download')).toBeVisible();
	await page.keyboard.press('Escape');

	// --- wipe ---------------------------------------------------------------
	await resetPatch(page);
	expect(await nodes(page)).toHaveLength(0);

	// --- back in -------------------------------------------------------------
	await page.getByTestId('topbar-load').click();
	await page.getByTestId('fs-upload').waitFor();
	// The input is hidden behind the button by design (a bare file input cannot be a ui primitive),
	// so the file is set on it directly — which is also the one part a unit test cannot reach.
	await page.locator('input[type=file]').setInputFiles(saved);

	await expect
		.poll(async () => (await nodes(page)).map((n) => n.type).sort())
		.toEqual(['Buffer', 'Oscillator']);

	// An uploaded patch has no home on the SERVER — the file it came from is on the user's own
	// machine, and the staged copy the backend read is already deleted. Adopting it would aim the
	// next silent Ctrl-S at a path that no longer exists.
	expect(await page.evaluate(() => (window as any).goofi.query.graph().savePath)).toBe(null);
});
