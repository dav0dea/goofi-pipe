import { test, expect, type Locator, type Page } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import { addNode, nodes, nodeParams, updateParam, waitForNode, waitForNoNode } from '../lib/goofi';
import { openSaveAs } from '../lib/topbar';

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
