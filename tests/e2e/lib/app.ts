import { expect, type Page } from '@playwright/test';

/** The bare readiness gate: `window.goofi` is published (AppShell mounted) and the node catalog
 * has arrived over the control WS (`query.nodeTypes()` non-empty ⇒ `hello` landed). On its own
 * this is only for a RELOAD mid-spec, where the spec's own state is legitimately still live —
 * everything else enters through `waitForApp`, which adds the hermeticity backstop. */
export async function appReady(page: Page): Promise<void> {
	await page.waitForFunction(
		() => {
			const g = (window as any).goofi;
			return !!g && (g.query.nodeTypes()?.length ?? 0) > 0;
		},
		undefined,
		{ timeout: 20_000 }
	);
}

/** Resolve once the app is fully live AND the shared backend is pristine — the readiness gate
 * every spec waits on before driving the façade. */
export async function waitForApp(page: Page): Promise<void> {
	await appReady(page);
	await expectPristineWorkspace(page);
}

/**
 * The hermeticity backstop, and it lives here because the readiness gate is the one line every
 * product spec runs before it touches anything.
 *
 * One backend serves every spec on this worker (`globalSetup.ts` spawns one per worker slot, and
 * `fullyParallel: false` keeps a worker's files running one after another) while AppShell pushes the
 * layout into the RUNNING PATCH on a 400ms debounce that can outlive the page. So a spec that splits
 * a panel or adds a tab and leaves early persists that arrangement, `hello` echoes it, and every
 * later spec on that worker boots into it. This suite has been bitten by it four times, and
 * every time the red landed on an innocent file measuring geometry — never on the file that leaked.
 *
 * Asserting it at ENTRY is what changes that: the failure names the cause, in the first spec after
 * the leak, instead of surfacing as a wrong bounding box in another project. A handful of reads
 * against an already-loaded page; nothing to do in the normal case. A spec that means to leave the
 * arrangement changed does not exist — the rule is that it hands the workspace back in a `finally`.
 *
 * What it deliberately does NOT check is the DIRTY flag. Editing anything sets it, so asserting it
 * clean would force a patch reset into all 37 specs that touch the graph — and a universal reset
 * would take the node guard above with it, since a reset can never leave a node behind to find.
 * The four globals below are all leaks a spec chooses to make; `unsaved_changes` is a byproduct of
 * doing any work at all.
 */
export async function expectPristineWorkspace(page: Page): Promise<void> {
	// The app is already live, so the default arrangement is on screen: a short budget keeps a leak
	// cheap to report instead of paying the 10s default on every spec after the one that leaked.
	const settled = { timeout: 2_000 };
	await expect(
		page.getByTestId('workspace-tabs').locator('.ui-tab'),
		'a previous spec left an extra workspace tab behind — hand it back in a `finally`'
	).toHaveCount(1, settled);
	await expect(
		page.locator('.panel'),
		'a previous spec left a split panel behind — hand it back in a `finally`'
	).toHaveCount(1, settled);
	// The third leakable global: GRAPH NODES. A leaked node is worse than a leaked panel — every
	// later page renders its viewer and subscribes its stream, so it surfaced as an fps reading
	// no single viewer could produce (viewer-fps-cap read 364 against a 30 cap: a dozen leaked
	// viewers summed), red on the innocent file as ever. Unlike its two siblings this is an
	// ABSENCE, and the leaked nodes arrive on the binary CRDT channel a beat after the JSON
	// `hello` that `appReady` gates on — a count read in that gap sees an empty replica and the
	// guard becomes a coin flip. `docSynced` is the positive signal that the replica has pulled;
	// only after it does an empty node list mean an empty graph. Read from the store (synchronous
	// with the doc), naming the leaked nodes so the red points at what was left, not just that
	// something was.
	await page.waitForFunction(() => (window as any).goofi.query.docSynced(), undefined, {
		timeout: 2_000
	});
	const leaked = await page.evaluate(() =>
		(window as any).goofi.query.graph().nodes.map((n: { type: string; uid: string }) => `${n.type}(${n.uid})`)
	);
	expect(leaked, 'a previous spec left graph nodes behind — remove them in a `finally`').toEqual([]);
	// The fourth leakable global, and the newest: the SAVE PATH. It is manager-owned since W, so it
	// outlives the page that wrote it and reaches every later spec through `hello` — and a named
	// patch changes what the app DOES, not just what it shows: `AppShell.triggerSave` overwrites
	// silently instead of opening the file browser. That is what took eleven specs red the first
	// time, none of them about saving. Unlike the dirty flag (a benign byproduct of any spec that
	// edits anything), naming the patch is always deliberate — so it is always someone's `finally`
	// to undo, with `resetPatch`.
	const savePath = await page.evaluate(() => (window as any).goofi.query.graph().savePath);
	expect(
		savePath,
		'a previous spec left the patch NAMED — hand it back with `resetPatch` in a `finally`'
	).toBe(null);
}

/**
 * Hand the shared backend back an empty, unnamed, clean patch — one manager transaction (`new`).
 *
 * The `finally` half of the save-path contract `expectPristineWorkspace` enforces at the other end,
 * and the successor to the `loadText(<the file we just saved>)` trick five specs used to play: a
 * `.gfi` is a zip archive now, so the façade carries no content-load door left to abuse for a reset.
 */
export async function resetPatch(page: Page): Promise<void> {
	await page.evaluate(() => (window as any).goofi.commands.newPatch());
	await expect
		.poll(() => page.evaluate(() => (window as any).goofi.query.graph().savePath))
		.toBe(null);
	await expect
		.poll(() => page.evaluate(() => (window as any).goofi.query.graph().nodes.length))
		.toBe(0);
}

/** Hand back a tab the spec added: close the last one. The close IS the command — there is no
 * debounced push to outwait any more — so the arrangement the manager keeps is pristine as soon as
 * the tab strip says so. The `finally` half of the contract `expectPristineWorkspace` enforces at
 * the other end. */
export async function closeAddedTab(page: Page): Promise<void> {
	const tabs = page.getByTestId('workspace-tabs');
	await tabs.getByRole('button', { name: 'Close tab' }).last().click();
	await expect(tabs.locator('.ui-tab'), 'the workspace is back to one tab').toHaveCount(1);
}

/** Split the sole default panel to the right, through the real header context menu. Two specs need
 *  a two-panel workspace — `panel-surface` to read the seam it paints, `inspector-orientation` to
 *  make the editor a narrow tall column — and a second copy is a second thing to keep true. */
export async function splitRight(page: Page): Promise<void> {
	const header = page.getByTestId('panel-header').first();
	await header.click({ button: 'right' });
	const item = page.locator('.context-menu .item', { hasText: 'Split Right' }).first();
	await expect(item).toBeVisible();
	await item.click();
	await expect(page.locator('.panel')).toHaveCount(2);
}

/**
 * Put the workspace back, through the split panel's own ✕. A split is a command against the RUNNING
 * PATCH — one backend per worker — so a spec that splits and leaves early persists a 2-panel
 * workspace that every later spec there boots into. It passes alone and depends on nothing but
 * screenshot latency, which is why it must be a `finally`.
 */
export async function closeSplit(page: Page): Promise<void> {
	await page.getByTestId('panel-header').nth(1).getByRole('button', { name: 'Close panel' }).click();
	await expect(page.locator('.panel'), 'the workspace is back to one panel').toHaveCount(1);
}
