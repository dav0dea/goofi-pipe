// The patch as a file: browsing for one, saving and loading it, the globals it carries, and
// what the console says while it runs.

import { test, expect, type Locator, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { settledBox } from '../lib/geometry';
import { waitForApp, resetPatch, splitRight, closeAddedTab } from '../lib/app';
import {
	addNode,
	nodes,
	nodeParams,
	updateParam,
	waitForNode,
	waitForNoNode,
	addGlobal,
	globals,
	addErroringNode
} from '../lib/goofi';
import { openSaveAs } from '../lib/topbar';

test.describe('the file browser', () => {
	/**
	 * The file browser is the app's ONLY persistence UI — Save/Load of `.gfi` patches — and it had
	 * zero e2e coverage. This is the safety net, written against the CURRENT hand-rolled modal so the
	 * M-Task-6b rewrite onto the `Dialog` primitive (its own overlay, its own `svelte:window` Escape
	 * handling, its own z-index) is provable rather than merely plausible.
	 *
	 * Two halves:
	 *  1. A REAL round trip — build a graph, Save it through the TopBar affordance to a real file on
	 *     the backend's disk, wipe the graph, Load the file back through the UI, and assert the nodes,
	 *     the link AND an edited param value came back. Not "a dialog closed".
	 *  2. The dismissal/navigation behaviours the Dialog swap puts at risk (Escape, outside click, the
	 *     ✕ button, filename retention, path-bar + up navigation), each pinned separately.
	 *
	 * Everything is driven through the REAL UI, not the `window.goofi` façade, because the UI is what
	 * 6b changes. The façade is used only to arrange and read back graph state.
	 *
	 * Hermeticity: all writes land in a per-run temp directory created here and removed in afterAll,
	 * so the file the round trip saves can never leak into a later run (the e2e backend serves the
	 * real, unjailed filesystem — see backend/goofi-bridge/src/fsbrowse.rs).
	 */

	/** Per-run scratch directory on the BACKEND's filesystem — same machine, since `globalSetup`
	 * spawns the fleet locally. `realpath` because the backend canonicalizes every path it echoes
	 * (`fsbrowse::normalize`), and the path bar is compared by string equality. */
	let scratch = '';
	/** A subdirectory, so the path-bar/up-navigation test has real structure to walk. */
	let nested = '';
	/** Unique per run — the round-trip patch's basename (no extension). */
	const patchName = `e2e-roundtrip-${process.pid}-${Date.now()}`;

	test.beforeAll(() => {
		scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-fs-')));
		nested = path.join(scratch, 'nested');
		fs.mkdirSync(nested);
	});

	test.afterAll(() => {
		fs.rmSync(scratch, { recursive: true, force: true });
	});

	/** The modal root. Present only while it is open — `AppShell` mounts it under `{#if fsMode}`. */
	function browser(page: Page): Locator {
		return page.getByTestId('fs-browser');
	}

	/**
	 * Open the browser the way a user does: the TopBar button.
	 *
	 * Save only reaches the modal while the patch has no home on disk — a named patch overwrites
	 * silently. That used to be free ("a fresh page always starts unnamed"), and it is not: the save
	 * path is MANAGER-owned since W, so it outlives the page and reaches every later spec. What makes
	 * this door dependable now is the contract at the other end — `expectPristineWorkspace` asserts a
	 * null path at entry, and any spec that names the patch hands it back with `resetPatch`. Reaching
	 * the modal over a NAMED patch is Save As's job (`lib/topbar.ts`'s `openSaveAs`).
	 */
	async function openBrowser(page: Page, mode: 'save' | 'load'): Promise<Locator> {
		await page.getByTestId(mode === 'save' ? 'topbar-save' : 'topbar-load').click();
		const modal = browser(page);
		await expect(modal, `the ${mode} browser opened`).toBeVisible();
		await expect(modal, 'the modal announces its mode').toHaveAttribute(
			'aria-label',
			mode === 'save' ? 'Save patch' : 'Load patch'
		);
		// The modal renders before its first `listDir` lands — the path bar is empty until it does.
		// Wait for that echo here so a test that navigates immediately isn't racing it.
		await expect(modal.getByTestId('fs-path-input'), 'the initial listing landed').not.toHaveValue('');
		return modal;
	}

	/** Navigate the path bar to `dir` and wait for the listing to land there. */
	async function navigateTo(page: Page, modal: Locator, dir: string): Promise<void> {
		const bar = modal.getByTestId('fs-path-input');
		await bar.fill(dir);
		await bar.press('Enter');
		// `go()` echoes the server-normalized path back into the bar — that is the arrival signal.
		await expect(bar, `the browser is showing ${dir}`).toHaveValue(dir);
	}

	/** Every node currently in the backend graph, removed. The e2e backend is shared across specs, so
	 * the round trip both starts and ends from a known-empty graph. */
	async function clearGraph(page: Page): Promise<void> {
		const uids = (await nodes(page)).map((n) => n.uid);
		if (uids.length) await page.evaluate((u) => (window as any).goofi.commands.removeNodes(u), uids);
		await expect.poll(async () => (await nodes(page)).length).toBe(0);
	}

	function links(page: Page): Promise<Array<Record<string, string>>> {
		return page.evaluate(() => (window as any).goofi.query.graph().links);
	}

	test('saves a patch through the UI and loads it back, restoring the real graph', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page); // …which is itself the assertion that the graph is empty and unnamed.
		try {
			// --- a recognisable patch: two linked nodes and one edited param -------------------------
			const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, osc);
			const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
			await waitForNode(page, buf);
			await page.evaluate(
				([o, b]) =>
					(window as any).goofi.commands.addLink({
						node_out: o,
						slot_out: 'out',
						node_in: b,
						slot_in: 'data'
					}),
				[osc, buf] as const
			);
			await expect.poll(async () => (await links(page)).length).toBe(1);
			// A value no default would produce, so "the patch came back" cannot pass by accident.
			await updateParam(page, osc, 'oscillator', 'amplitude', 0.4242);
			await expect
				.poll(async () => (await nodeParams(page, osc))?.oscillator?.amplitude?.value)
				.toBeCloseTo(0.4242, 5);
			// Names are minted by the manager and persisted into the .gfi — capture them to compare.
			const savedNames = (await nodes(page)).map((n) => n.name).sort();

			// --- SAVE through the real modal ---------------------------------------------------------
			const saveModal = await openBrowser(page, 'save');
			await navigateTo(page, saveModal, scratch);
			await expect(saveModal.getByTestId('fs-list'), 'the fresh scratch dir lists its subdir').toContainText(
				'nested'
			);
			await saveModal.getByTestId('fs-filename').fill(patchName);
			await saveModal.getByTestId('fs-save').click();
			await expect(saveModal, 'confirming Save closes the browser').toBeHidden();

			// The file really exists on the backend's disk, and the app now knows where the patch lives.
			const patchFile = path.join(scratch, `${patchName}.gfi`);
			await expect.poll(() => fs.existsSync(patchFile), { message: 'the .gfi landed on disk' }).toBe(true);
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
				.toBe(patchFile);
			// A backend write clears the dirty flag (bridge `save` → `set_dirty(false)`).
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.graph().unsavedChanges))
				.toBe(false);

			// --- wipe, then LOAD it back through the real modal ---------------------------------------
			await clearGraph(page);
			expect(await links(page), 'removing the nodes took the link with them').toEqual([]);

			const loadModal = await openBrowser(page, 'load');
			// The load browser opens beside the current patch (`initialPath = dirOf(savePath)`), so no
			// navigation is needed — assert that rather than papering over it by re-typing the path.
			await expect(
				loadModal.getByTestId('fs-path-input'),
				'the browser opens in the directory the patch lives in'
			).toHaveValue(scratch);
			// Open is disabled until a .gfi is selected; a single click selects it.
			await expect(loadModal.getByTestId('fs-open'), 'Open is inert with nothing selected').toBeDisabled();
			await loadModal.getByTestId('fs-entry').filter({ hasText: `${patchName}.gfi` }).click();
			await expect(loadModal.getByTestId('fs-open'), 'selecting a .gfi arms Open').toBeEnabled();
			await loadModal.getByTestId('fs-open').click();
			await expect(loadModal, 'confirming Open closes the browser').toBeHidden();

			// --- the graph is genuinely restored -------------------------------------------------------
			await expect.poll(async () => (await nodes(page)).length, { message: 'both nodes came back' }).toBe(2);
			const restored = await nodes(page);
			expect(restored.map((n) => n.type).sort()).toEqual(['Buffer', 'Oscillator']);
			expect(restored.map((n) => n.name).sort()).toEqual(savedNames);

			const rOsc = restored.find((n) => n.type === 'Oscillator')!;
			const rBuf = restored.find((n) => n.type === 'Buffer')!;
			await expect.poll(async () => (await links(page)).length, { message: 'the link came back' }).toBe(1);
			expect((await links(page))[0]).toMatchObject({
				node_out: rOsc.uid,
				slot_out: 'out',
				node_in: rBuf.uid,
				slot_in: 'data'
			});
			// The edited param survived the round trip — the strongest signal that this is the saved patch
			// and not a coincidentally-shaped fresh one.
			await expect
				.poll(async () => (await nodeParams(page, rOsc.uid))?.oscillator?.amplitude?.value)
				.toBeCloseTo(0.4242, 5);
		} finally {
			// This test both SAVES and LOADS, so it names the patch twice — and it is the file the
			// hermeticity guard caught leaking: the `clearGraph` that used to sit here ran only on the
			// happy path, so an abort above it left two nodes for nineteen later specs to trip over.
			// A `finally` is the whole fix, and `resetPatch` is the reset that covers the name too.
			await resetPatch(page);
		}
	});

	test.describe('modal dismissal (the behaviours M-Task 6b re-homes onto Dialog)', () => {
		test('Escape closes the browser', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const modal = await openBrowser(page, 'save');
			await page.keyboard.press('Escape');
			await expect(modal, 'Escape dismisses the modal').toBeHidden();
		});

		test('a click outside the modal closes the browser', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const modal = await openBrowser(page, 'load');
			// The modal is centred and size-capped, so the viewport's top-left corner is backdrop.
			// Assert that before clicking, rather than trusting the layout.
			const box = (await modal.boundingBox())!;
			expect(box.x, 'the (4,4) click point is left of the modal').toBeGreaterThan(8);
			expect(box.y, 'the (4,4) click point is above the modal').toBeGreaterThan(8);
			await page.mouse.click(4, 4);
			await expect(modal, 'an outside click dismisses the modal').toBeHidden();
		});

		test('the ✕ button closes the browser', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const modal = await openBrowser(page, 'save');
			await modal.getByRole('button', { name: 'Close' }).click();
			await expect(modal, 'the header close button dismisses the modal').toBeHidden();
		});

		// CHARACTERIZATION — not a requirement, a record of what today does. `AppShell` mounts the
		// browser under `{#if fsMode}`, so dismissing it UNMOUNTS the component and the typed filename
		// dies with it; reopening re-seeds from `suggestedName`, which is '' while the patch is
		// unnamed. 6b's Dialog swap must decide this deliberately (keep the draft or keep discarding
		// it) instead of changing it by accident — if it flips, this test is the alarm.
		test('a typed filename is DISCARDED when the browser is dismissed', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const modal = await openBrowser(page, 'save');
			await modal.getByTestId('fs-filename').fill('discard-probe');
			await page.keyboard.press('Escape');
			await expect(modal).toBeHidden();

			const reopened = await openBrowser(page, 'save');
			await expect(
				reopened.getByTestId('fs-filename'),
				'today: the draft filename does not survive a dismissal'
			).toHaveValue('');
		});
	});

	/**
	 * The browser is a MODAL, so while it owns the screen the app's global chords must stand down.
	 * They did not: `undoKeyAction`'s typing guard only covers INPUT/TEXTAREA/SELECT, and FsBrowser
	 * never raised `ui().modalOpen` — so a Ctrl+Z after clicking a file row (a <button>) fell straight
	 * through to the graph history and undid a command behind the modal (M-Task 6a, C29). Ctrl+S/Ctrl+O
	 * had the same hole: AppShell re-ran its handler with the browser already up (C31).
	 */
	test.describe('global shortcuts stand down while the browser owns the screen', () => {
		const historyLength = (page: Page): Promise<number> =>
			page.evaluate(() => (window as any).goofi.query.historyLength());

		test('Ctrl+Z does not reach the graph behind the modal', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			await clearGraph(page);

			const uid = await addNode(page, 'Buffer', 'signal');
			await waitForNode(page, uid);
			const before = await historyLength(page);

			const modal = await openBrowser(page, 'load');
			await navigateTo(page, modal, scratch);
			// Click a file row to put focus on a <button> — the case the typing guard does NOT cover.
			await modal.getByTestId('fs-entry').filter({ hasText: 'nested' }).click();
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.modalOpen()), {
					message: 'the open browser holds the global standdown'
				})
				.toBe(true);

			await page.keyboard.press('Control+z');
			// Proving a NON-event needs a settle window: an undo is a WS round trip to a backend on
			// localhost (single-digit ms), so 400ms is orders of magnitude more than it would need.
			await page.waitForTimeout(400);
			expect(await historyLength(page), 'no undo was consumed behind the modal').toBe(before);
			expect(
				(await nodes(page)).map((n) => n.uid),
				'the graph is untouched while the browser is open'
			).toContain(uid);

			// Positive control: the SAME chord undoes once the browser is dismissed — so the assertion
			// above is about the standdown, not about the key never reaching the app at all.
			await page.keyboard.press('Escape');
			await expect(modal).toBeHidden();
			await page.keyboard.press('Control+z');
			await waitForNoNode(page, uid);

			await clearGraph(page);
		});

		test('Ctrl+S does not re-trigger Save while the browser is already open', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const modal = await openBrowser(page, 'load');
			await page.keyboard.press('Control+s');
			// Before the fix AppShell ran triggerSave() again and flipped the open browser into save mode
			// (and, on a named patch, would have written the file behind it).
			await page.waitForTimeout(400);
			await expect(modal, 'the Load browser stayed the Load browser').toHaveAttribute(
				'aria-label',
				'Load patch'
			);
			await page.keyboard.press('Escape');
			await expect(modal).toBeHidden();
		});

		test('Ctrl+S / Ctrl+O are still swallowed while the standdown is up', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const modal = await openBrowser(page, 'load');
			// A window listener registered AFTER AppShell's runs second on the same target/phase, so it
			// observes whether the app called preventDefault(). Standing down from ACTING on the chord is
			// not the same as letting it through: unprevented, Chrome runs its own Save-page / Open-file
			// accelerator over the app (invisible headless, which is why this shipped).
			await page.evaluate(() => {
				(window as any).__chords = [] as { key: string; prevented: boolean }[];
				window.addEventListener('keydown', (e) => {
					const k = e.key.toLowerCase();
					if ((e.ctrlKey || e.metaKey) && (k === 's' || k === 'o'))
						(window as any).__chords.push({ key: k, prevented: e.defaultPrevented });
				});
			});

			await page.keyboard.press('Control+s');
			await page.keyboard.press('Control+o');
			expect(
				await page.evaluate(() => (window as any).__chords),
				'both app chords were consumed rather than falling through to the browser'
			).toEqual([
				{ key: 's', prevented: true },
				{ key: 'o', prevented: true }
			]);

			await page.keyboard.press('Escape');
			await expect(modal).toBeHidden();
		});
	});

	test('the path bar and the up button navigate the backend filesystem', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const modal = await openBrowser(page, 'load');
		await navigateTo(page, modal, scratch);

		// Double-click descends into a directory (a single click only highlights).
		const dir = modal.getByTestId('fs-entry').filter({ hasText: 'nested' });
		await expect(dir, 'the subdirectory is listed').toBeVisible();
		await dir.dblclick();
		await expect(modal.getByTestId('fs-path-input'), 'descended into the subdirectory').toHaveValue(nested);
		await expect(modal.getByTestId('fs-list'), 'the empty subdirectory says so').toContainText(
			'Empty folder.'
		);

		// …and the up button climbs back out.
		await modal.getByTitle('Up one level').click();
		await expect(modal.getByTestId('fs-path-input'), 'up one level returns to the parent').toHaveValue(
			scratch
		);
		await expect(dir, 'the parent listing is back').toBeVisible();
	});

	/**
	 * The filename field is the one control in this dialog with a real minimum, and it is DESKTOP
	 * geometry — R was not supposed to move it. `1ca82f4` changed it from a fixed `14rem` to
	 * `flex: 1 1 8rem` as a side effect of making the footer wrap at 320px, and the grow can never
	 * fire: `.ui-bar-group` declares no `flex` (so `0 1 auto`) while `.ui-bar-spacer` owns all of the
	 * bar's slack, so the field resolved at its `size=20` max-content contribution — about 154px
	 * against the 196px it had. The wrap alone is what fixed the phone; the basis is the desktop's.
	 */
	test('the save dialog keeps its full-width filename field', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const modal = await openBrowser(page, 'save');
		const field = modal.getByTestId('fs-filename');
		const { width, rem } = await field.evaluate((el) => ({
			width: el.getBoundingClientRect().width,
			rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
		}));
		// Stated in rem, since the root size is a responsive clamp: 14rem is the pre-R fixed width.
		expect(width, 'the field is 14rem wide, as it was before R').toBeGreaterThanOrEqual(14 * rem - 1);
		await page.keyboard.press('Escape');
		await expect(modal).toBeHidden();
	});

	/**
	 * `window.goofi.commands.save(path)` is the seam the whole committed suite drives, and both doors
	 * onto a save now learn the patch's home the same way: from the MANAGER. There is no client-side
	 * latch left — the `save` arm publishes `save_path_changed` and the snapshot carries a real path —
	 * which is what makes the name survive a reload and reach a second tab (C38).
	 *
	 * The tail is the other half of that: once the patch is named, a plain Save overwrites in silence
	 * and **Save As is the only door back to the browser**. It is also the only door at a phone width,
	 * where the caret is the first action to spill — hence one testid on the row and two ways in.
	 */
	test('saving through the automation façade names the patch, exactly as the header does', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const target = path.join(scratch, `${patchName}-facade.gfi`);
		try {
			await page.evaluate((p) => (window as any).goofi.commands.save(p), target);
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath), {
					message: 'the patch has a home on disk and the client knows it'
				})
				.toBe(target);
			// …and the header agrees, which is what "named" means to a user: a plain Save from here
			// overwrites silently instead of re-opening the browser.
			await expect(page.locator('.topbar .path')).toHaveText(new RegExp(`${patchName}-facade\\.gfi$`));
			await page.getByTestId('topbar-save').click();
			await expect(browser(page), 'a named patch saves without asking again').toBeHidden();

			// Save As still reaches it — the re-home door, and the only one a named patch has.
			await openSaveAs(page);
			await expect(browser(page), 'Save As opens the browser over a named patch').toBeVisible();
			await page.keyboard.press('Escape');
			await expect(browser(page)).toBeHidden();
		} finally {
			fs.rmSync(target, { force: true });
			await resetPatch(page);
		}
	});
});

test.describe('saving and loading a patch', () => {
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
});

test.describe('the patch remembers its name', () => {
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

	test('the client learns the patch\u2019s name from the manager, on save and on load', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		try {
			const target = path.join(scratch, 'roundtrip.gfi');
			const uid = await addNode(page, 'Oscillator');
			await waitForNode(page, uid);

			await page.evaluate((p) => (window as any).goofi.commands.save(p), target);
			// The store learned the path from the MANAGER, not from a client-side write — which is the
			// whole of C38 on this side of the wire. That the archive is a real zip holding a real
			// patch is the Rust suite's (`session.rs`, `browser.rs`).
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
				.toBe(target);

			await addNode(page, 'Buffer', 'signal'); // diverge
			await page.evaluate((p) => (window as any).goofi.commands.load(p), target);
			await expect.poll(async () => (await nodes(page)).map((n) => n.type)).toEqual(['Oscillator']);
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
			// A patch has a graph, a file AND an arrangement; New must inherit none of the three. The
			// arrangement is the one that survived the manager fix: `graph_replaced` carries a null
			// layout on the SAME instance id, so the client used to take neither the hydrate branch nor
			// the fresh-session reset and left the previous patch's panels on screen — and then pushed
			// them back down as the new patch's stored layout on the next split.
			await splitRight(page);
			await page.evaluate((p) => (window as any).goofi.commands.save(p), path.join(scratch, 'named.gfi'));
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
				.not.toBe(null);

			await page.evaluate(() => (window as any).goofi.commands.newPatch());
			await expect.poll(async () => (await nodes(page)).length).toBe(0);
			await expect(page.locator('.panel'), 'a New patch opens on the default arrangement').toHaveCount(
				1
			);
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
});

test.describe('a patch that carries its own nodes', () => {
	/**
	 * A patch's own `nodes/` directory, end to end against the REAL discovery seam — the probe, the
	 * tier routing and the registry the shipped `nodes/` tree goes through, not a stub. That is the
	 * whole claim of W1 ("loaded and managed in the exact same way, using the same code"), and nothing
	 * below the e2e can prove it: every Rust test injects a stub scan precisely so it needs no
	 * interpreter.
	 *
	 * The file is written straight into the live workspace mount, which is what an agent in the harness
	 * will do. Its path is not derivable — a per-run temp directory under a random nonce — so the
	 * façade's `openWorkspace` is the only way in, and that is the same door the agent gets.
	 *
	 * Hermeticity: the node file is removed and a final rescan re-derives the registry, so the shared
	 * backend hands back the palette it started with.
	 */

	const source = (cls: string): string => `import goofi


class ${cls}(goofi.Node):
    """A node that came with the patch."""

    OUTPUTS = {"out": goofi.DataType.ARRAY}

    def process(self):
        return [1.0]
`;

	/** The palette row for `type`, as the add menu reads it. */
	function paletteRow(page: Page, type: string): Promise<{ type: string; source: string } | null> {
		return page.evaluate((t) => {
			const types = (window as any).goofi.query.nodeTypes() ?? [];
			return types.find((x: { type: string }) => x.type === t) ?? null;
		}, type);
	}

	function rescan(page: Page): Promise<{ added: string[]; changed: string[]; removed: string[] }> {
		return page.evaluate(() => (window as any).goofi.commands.rescanNodes());
	}

	test('a node file written into the patch workspace joins the palette, marked as the patch’s own', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		const workspace: string = await page.evaluate(() =>
			(window as any).goofi.commands.openWorkspace()
		);
		const dir = path.join(workspace, 'nodes');
		const file = path.join(dir, 'e2e_patch_node.py');
		fs.mkdirSync(dir, { recursive: true });

		try {
			expect(await paletteRow(page, 'E2ePatchNode'), 'not there before it is written').toBeNull();

			fs.writeFileSync(file, source('E2ePatchNode'));
			const diff = await rescan(page);
			// ONLY the new file — the shipped `nodes/` tree was indexed by the boot scan, which runs
			// this very function, so a refresh reports what changed rather than re-announcing goofi's
			// own nodes as new.
			expect(diff.added, 'the rescan reports exactly what it found').toEqual(['E2ePatchNode']);

			// The catalog reaches this tab through the `node_types` broadcast, not a reply, so wait for
			// the store rather than asserting on the round trip that produced it.
			await page.waitForFunction(
				() =>
					((window as any).goofi.query.nodeTypes() ?? []).some(
						(t: { type: string }) => t.type === 'E2ePatchNode'
					),
				undefined,
				{ timeout: 20_000 }
			);
			const row = await paletteRow(page, 'E2ePatchNode');
			expect(row?.source, 'and the palette says where it came from').toBe('patch');

			// A shipped node is the control: same palette, other provenance.
			expect((await paletteRow(page, 'Oscillator'))?.source).toBe('builtin');

			// It is a real type, not merely a row: it instantiates through the ordinary add path.
			const uid: string = await page.evaluate(() =>
				(window as any).goofi.commands.addNode('E2ePatchNode', 'python', [0, 0])
			);
			expect(uid).toBeTruthy();
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		} finally {
			fs.rmSync(file, { force: true });
			const gone = await rescan(page);
			expect(gone.removed, 'the palette is handed back as it was found').toContain('E2ePatchNode');
		}
	});

	test('the palette’s own refresh button rescans and says what moved', async ({ page }) => {
		// The test above drives `rescanNodes` through the façade, which the BUTTON is not: it could be
		// deleted, mis-wired or put behind `:hover` (D-R7) with the whole suite still green. This is the
		// only door a human has onto a file they just wrote, so it is driven as a human drives it.
		await page.goto('/');
		await waitForApp(page);

		const workspace: string = await page.evaluate(() =>
			(window as any).goofi.commands.openWorkspace()
		);
		const dir = path.join(workspace, 'nodes');
		const file = path.join(dir, 'e2e_button_node.py');
		fs.mkdirSync(dir, { recursive: true });
		fs.writeFileSync(file, source('E2eButtonNode'));

		try {
			await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
			const button = page.getByTestId('add-menu-rescan');
			// Asserted before anything has hovered the menu: the refresh is not allowed to be revealed.
			await expect(button, 'the refresh is there for a pointer that never hovered').toBeVisible();

			await button.click();
			await expect(page.getByTestId('toast'), 'the button reported the diff it got back').toContainText(
				'1 added'
			);

			// And the row it added says where it came from — in the text a person reads, which is the
			// only place provenance is rendered.
			await expect(
				page.getByTestId('add-menu-list').locator('.item', { hasText: 'E2eButtonNode' })
			).toContainText('this patch');
			await page.keyboard.press('Escape');
		} finally {
			fs.rmSync(file, { force: true });
			await rescan(page);
		}
	});
});

test.describe('the globals panel', () => {
	/** Borrow the default node-editor panel as a Globals panel, run `body`, then give it back.
	 *
	 * The layout lives in the RUNNING PATCH and outlives this page (AppShell pushes it on a 400ms
	 * debounce), so a spec that leaves without restoring persists a globals-shaped workspace that
	 * every later spec boots into. Mirrors `console-rows.spec.ts`. */
	async function inGlobalsPanel(page: Page, body: () => Promise<void>): Promise<void> {
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'globals'), panelId);
		try {
			await expect(page.getByTestId('globals-panel')).toBeVisible();
			await body();
		} finally {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		}
	}

	/**
	 * The client's REPLICA of the globals, which is all this file can ask: that they arrive over the
	 * doc and carry the system flag the panel gates its rename and delete on. That a global adds,
	 * edits, renames and refuses a re-type is the Rust suite's (`editing.rs`, `session.rs`).
	 */
	test('the patch globals reach the client with their system flag', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await expect
			.poll(async () => (await globals(page)).find((g) => g.name === 'default_ufreq'))
			.toMatchObject({ system: true, type: 'float' });
		try {
			await addGlobal(page, 'subject', 'P07', 'string');
			await expect
				.poll(async () => (await globals(page)).find((g) => g.name === 'subject'))
				.toMatchObject({ system: false, value: 'P07' });
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.removeGlobal('subject'));
		}
	});

	test('the Globals panel renders when opened', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// This one opens the panel in a NEW TAB rather than borrowing the editor, so the tab is what
		// has to be handed back — see `closeAddedTab`.
		await page.evaluate(() => (window as any).goofi.commands.addTab('globals'));
		try {
			await expect(page.getByTestId('globals-panel')).toBeVisible();
		} finally {
			await closeAddedTab(page);
		}
	});

	// A global's VALUE is machine-read: expressions resolve `globals.<name>` and node process()/setup()
	// read `ctx.globals`. It must not be autocorrected, autocapitalised or red-underlined as prose. The
	// panel's author already declared exactly that one row up, on the NAME field — the `TextInput`
	// migration dropped it from the value cell, where the primitive's `text` default supplies the
	// opposite (spellcheck on, autocorrect on, sentence capitalisation) and a consumer cannot override
	// it by spreading, because the mode attributes are applied AFTER `{...rest}`.
	test('a string global’s value cell is typed as machine-read, not prose', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await addGlobal(page, 'm15_token', 'AF7-AF8', 'string');
		try {
			await inGlobalsPanel(page, async () => {
				const cell = page.locator('tr[data-name="m15_token"]').getByTestId('global-value');
				await expect(cell, 'the value round-tripped into the panel').toHaveValue('AF7-AF8');
				await expect(cell).toHaveAttribute('spellcheck', 'false');
				await expect(cell).toHaveAttribute('autocorrect', 'off');
				await expect(cell).toHaveAttribute('autocapitalize', 'off');
				await expect(cell).toHaveAttribute('autocomplete', 'off');
			});
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.removeGlobal('m15_token'));
		}
	});

	// The panel's own commit path, which `d0f31b1` rebuilt onto the `useLiveValue` latch and which no
	// test has crossed since: globals.spec drives values through the `window.goofi` façade, which
	// bypasses the widgets entirely. Typing is the part that can break — the latch has to hold the
	// buffer while the doc echoes back, and the panel has to parse to the global's DECLARED type
	// rather than to whatever the widget handed it.
	test('the Globals panel commits a typed number as the global’s declared type', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await addGlobal(page, 'm18_gain', 0, 'float');
		await addGlobal(page, 'm18_count', 0, 'int');
		try {
			await inGlobalsPanel(page, async () => {
				const gain = page.locator('tr[data-name="m18_gain"]').getByTestId('global-value');
				await gain.fill('2.5');
				await gain.blur();
				await expect
					.poll(async () => (await globals(page)).find((g) => g.name === 'm18_gain')?.value)
					.toBeCloseTo(2.5, 5);

				// An int global rounds on commit — the panel parses to the declared type, and 7.6 typed
				// into an int is 8, not a float that silently retypes the global.
				const count = page.locator('tr[data-name="m18_count"]').getByTestId('global-value');
				await count.fill('7.6');
				await count.blur();
				await expect
					.poll(async () => (await globals(page)).find((g) => g.name === 'm18_count')?.value)
					.toBe(8);
			});
		} finally {
			await page.evaluate(() => {
				const c = (window as any).goofi.commands;
				return Promise.all([c.removeGlobal('m18_gain'), c.removeGlobal('m18_count')]);
			});
		}
	});

	// A global's name AND its value are expression identifiers and machine-read data (spec D-T3), so
	// both read in mono — including the EDITABLE cells, which is where the two-face flip could lose
	// them silently: the `ui` inputs carry `font: inherit` by design, so they render whatever the cell
	// hands down, and body-level sans would take them without changing a line in this file. The column
	// headers above them are chrome and stay sans, which is what makes this a taxonomy assertion rather
	// than a "everything here is mono" one.
	test('a global’s editable name and value render mono, its column header sans (D-T3)', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		await addGlobal(page, 'm19_face', 'AF7', 'string');
		try {
			await inGlobalsPanel(page, async () => {
				const row = page.locator('tr[data-name="m19_face"]');
				const face = (loc: ReturnType<typeof row.locator>) =>
					loc.evaluate((el) => getComputedStyle(el).fontFamily.split(',')[0].replace(/["']/g, ''));
				expect(await face(row.getByTestId('global-name')), 'the name input').toBe('JetBrains Mono');
				expect(await face(row.getByTestId('global-value')), 'the value input').toBe('JetBrains Mono');
				expect(await face(page.locator('th.c-name')), 'the column header').toBe('Inter');
				// The add row sits OUTSIDE the table, so the cell seam above cannot reach it — and a name
				// being typed is the same identifier the cell will hold the moment Add is pressed.
				expect(await face(page.getByTestId('global-add-name')), 'the name being typed').toBe(
					'JetBrains Mono'
				);
			});
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.removeGlobal('m19_face'));
		}
	});
});

test.describe('the console', () => {
	/**
	 * The Console's virtual scroller keeps its own height model — `estimateH()` in the script mirrors
	 * `.row`'s padding (`PAD`) and its text line box (`LINE_H`) in px, because `layout.cum` sums an
	 * ESTIMATE for every row the ResizeObserver has not measured yet. When the model and the DOM
	 * disagree the thumb reads the wrong length and the content drifts as rows measure in.
	 *
	 * M's IconButton migration widened that gap: the per-row copy button is rendered unconditionally
	 * (only faded out), and the primitive floors its box at `--hit` = 28px on a fine pointer — four
	 * times the pre-existing 3px error, and enough to make a single-line row 33px against a 20px
	 * estimate. The assertions below are content-length independent: they pin the MODEL (a row is its
	 * text box plus the padding and border the estimator accounts for) rather than one row's pixels.
	 */
	test('a console row is sized by its text, not by its action buttons', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		// The default layout is one node editor; borrow it and give it straight back, so no later spec
		// inherits a console-shaped workspace.
		await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

		// A node erroring on its empty required input (see `addErroringNode`); the graph store mirrors
		// the error into the console as a stderr line. The cheapest real console content.
		const uid = await addErroringNode(page);
		try {
			const row = page.getByTestId('console-entry').first();
			await expect(row, 'the node error reached the console').toBeVisible();

			const m = await row.evaluate((el) => {
				const px = (sel: string) =>
					el.querySelector(sel)!.getBoundingClientRect().height;
				const cs = getComputedStyle(el);
				return {
					row: el.getBoundingClientRect().height,
					txt: px('.txt'),
					copy: px('.console-copy-btn'),
					pad: parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom),
					border: parseFloat(cs.borderBottomWidth) + parseFloat(cs.borderTopWidth)
				};
			});

			expect(m.copy, 'the copy button fits inside the row line box it shares').toBeLessThanOrEqual(
				m.txt
			);
			expect(m.row, 'the row is exactly text + padding + border — what estimateH models').toBeCloseTo(
				m.txt + m.pad + m.border,
				0
			);
			// And the estimator's own constants are the ones the DOM actually uses.
			expect(m.pad, 'PAD mirrors the row padding').toBe(4);
			expect(m.border, 'the estimate accounts for the row border').toBe(1);
			expect(m.txt % 16, 'the text box is a whole number of LINE_H lines').toBe(0);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			// The restore is a command against the RUNNING PATCH, which outlives this page — and the
			// assertion above only passes once the manager's delta drew it back, so it has landed.
		}
	});
});

test.describe('a node reports an error', () => {
	// M-Task 2 gave the ErrorPanel the real dismissal it never had: the error dropdown used to only
	// toggle from the chip — no Escape, no outside-click. It now delegates to the Popover primitive,
	// so this asserts the NEW behaviour on the REAL panel, driven by a REAL node error: an unconnected
	// required input slot on a ticking node (see `addErroringNode`). That sets the same `error` field
	// that drives the floating chip. The node is torn down after each test so the shared backend graph
	// stays clean for later specs.
	test.describe('ErrorPanel dismissal (delegated to Popover)', () => {
		let created: string[] = [];

		test.afterEach(async ({ page }) => {
			for (const uid of created) {
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
			created = [];
		});

		async function summonErrorChip(page: Page) {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addErroringNode(page); // its empty required input surfaces as the chip
			created.push(uid);
			const chipHost = page.getByTestId('error-chip');
			await expect(chipHost, 'a real node error raises the floating error chip').toBeVisible();
			return chipHost;
		}

		test('the chip opens the error popover, and Escape dismisses it', async ({ page }) => {
			const chipHost = await summonErrorChip(page);
			const popover = page.getByTestId('error-popover');
			const chip = chipHost.locator('button').first();

			await expect(popover, 'closed by default').toBeHidden();
			// The chip is a disclosure trigger, and says so — the surface it opens carries no popup role
			// by design (its children are plain buttons), so `aria-expanded` is the whole story it can tell.
			await expect(chip, 'the trigger reports its collapsed state').toHaveAttribute(
				'aria-expanded',
				'false'
			);
			await chip.click();
			await expect(popover, 'clicking the chip opens the popover').toBeVisible();
			await expect(chip, 'and its expanded one').toHaveAttribute('aria-expanded', 'true');

			await page.keyboard.press('Escape');
			await expect(popover, 'Escape dismisses the popover (new behaviour)').toBeHidden();
			await expect(chip, 'and back to collapsed').toHaveAttribute('aria-expanded', 'false');
		});

		test('an outside click dismisses the error popover', async ({ page }) => {
			const chipHost = await summonErrorChip(page);
			const popover = page.getByTestId('error-popover');

			await chipHost.locator('button').click();
			await expect(popover).toBeVisible();
			// Click the top-left of the viewport — neither the popover surface nor the chip anchor.
			await page.mouse.click(10, 10);
			await expect(popover, 'an outside click dismisses the popover (new behaviour)').toBeHidden();
		});

		// The chip is anchored to the BOTTOM of the editor (`bottom: 12px`), and the clamp only ever
		// shifted — never flipped — so the surface slid up until it fit and then ENCLOSED the chip on both
		// axes at a higher z-index. Two consequences: the chip's toggle-to-close (the pre-M panel's only
		// dismissal) became unreachable, and the click was not inert — it landed on the first error row and
		// focused a node the user never asked for. Pre-M this panel opened upward.
		test('the popover opens clear of its own chip, and the chip still toggles it closed', async ({
			page
		}) => {
			const chipHost = await summonErrorChip(page);
			const chip = chipHost.locator('button');
			const popover = page.getByTestId('error-popover');

			await chip.click();
			await expect(popover).toBeVisible();
			const chipBox = (await chip.boundingBox())!;
			// The Popover primitive positions the surface itself; measure the surface, not the list inside.
			const menuBox = (await page.locator('.ui-popover').boundingBox())!;
			const overlaps =
				chipBox.x < menuBox.x + menuBox.width &&
				menuBox.x < chipBox.x + chipBox.width &&
				chipBox.y < menuBox.y + menuBox.height &&
				menuBox.y < chipBox.y + chipBox.height;
			expect(overlaps, 'the popover does not cover the chip that opens it').toBe(false);

			const before = await page.evaluate(() => (window as any).goofi.query.selection());
			await chip.click(); // the toggle-to-close — must reach the chip, and must be inert otherwise
			await expect(popover, 'the chip toggles its own popover closed').toBeHidden();
			expect(
				await page.evaluate(() => (window as any).goofi.query.selection()),
				'closing the popover selected nothing behind it'
			).toEqual(before);
		});

		// M-5 restored the flip, but only as a SNAPSHOT: the Popover measures its surface exactly once
		// per open, so a node erroring while the list is already open grows the surface DOWNWARD from a
		// top pinned for the old height — back over the chip, at --z-menu over --z-chip. That is verbatim
		// the defect M-5 fixed, arriving half a second later: the toggle-to-close is buried and a click at
		// the chip's coordinates lands on a `.prow` and focuses a node the user never asked for. Pre-M the
		// panel anchored in pure CSS (`bottom: calc(100% + 6px)`), which self-corrected for ANY height;
		// the migration replaced that with a measured `top`, and M-5 restored only the initial placement.
		test('the popover stays clear of its chip when a second node errors while it is open', async ({
			page
		}) => {
			const chipHost = await summonErrorChip(page);
			const chip = chipHost.locator('button');
			const popover = page.locator('.ui-popover');
			const rows = page.locator('.error-list .prow');

			await chip.click();
			await expect(popover).toBeVisible();
			await expect(rows, 'one errored node so far').toHaveCount(1);

			// A second erroring node added WHILE the popover is open. `activeNodes` derives live from the
			// control plane, so the list grows under a surface that was measured for one row.
			const uid = await addErroringNode(page);
			created.push(uid);
			await expect(rows, 'the open list grew a row').toHaveCount(2);

			// SETTLED, both: a popover that has just grown a row RE-PLACES itself, and a box read
			// before that lands describes where it used to be — which reads as an overlap that is not
			// there, or hides one that is.
			const chipBox = await settledBox(chip);
			const menuBox = await settledBox(popover);
			const overlaps =
				chipBox.x < menuBox.x + menuBox.width &&
				menuBox.x < chipBox.x + chipBox.width &&
				chipBox.y < menuBox.y + menuBox.height &&
				menuBox.y < chipBox.y + chipBox.height;
			expect(overlaps, 'the GROWN popover still does not cover the chip that opens it').toBe(false);
		});

		// The row inside the popover is the one M kept bespoke here (a stacked name-over-message list
		// row, not an action). M-Task 7 strips app.css's base `button` SKIN, so it must render from its
		// own rule alone — including the fade on its hover fill, which it was inheriting from that skin
		// until the strip made the dependency visible. This is the only place the row is reachable.
		test('the kept-bespoke error row renders from its own rule, not the base skin', async ({
			page
		}) => {
			const chipHost = await summonErrorChip(page);
			await chipHost.locator('button').click();
			const row = page.locator('.error-list .prow').first();
			await expect(row).toBeVisible();
			// Park the cursor away from the list and poll until the hover fill has faded back out (the
			// row's own transition), or the probe samples a mid-fade colour rather than the rest state.
			await page.mouse.move(5, 5);
			await expect
				.poll(() => row.evaluate((el) => getComputedStyle(el).backgroundColor), {
					message: 'the row is transparent at rest'
				})
				.toBe('rgba(0, 0, 0, 0)');
			const s = await row.evaluate((el) => {
				const cs = getComputedStyle(el);
				return {
					fontFamily: cs.fontFamily,
					fontSize: parseFloat(cs.fontSize),
					background: cs.backgroundColor,
					borderWidth: cs.borderTopWidth,
					radius: cs.borderTopLeftRadius,
					padTop: parseFloat(cs.paddingTop),
					padLeft: parseFloat(cs.paddingLeft),
					transition: cs.transitionProperty,
					rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
				};
			});
			expect(s.fontFamily, 'the error row renders in the app mono face').toContain('JetBrains Mono');
			expect(s.fontSize, 'it inherits the popover surface size (`font: inherit`)').toBeCloseTo(
				0.82 * s.rem,
				0
			);
			expect(s.background, 'transparent at rest').toBe('rgba(0, 0, 0, 0)');
			expect(s.borderWidth, 'borderless').toBe('0px');
			expect(s.radius, 'its hover fill is rounded (--radius-sm)').toBe('4px');
			expect(s.padTop).toBeCloseTo(0.375 * s.rem, 0);
			expect(s.padLeft).toBeCloseTo(0.625 * s.rem, 0);
			expect(s.transition, 'the hover fill fades in').toContain('background');
		});
	});
});
