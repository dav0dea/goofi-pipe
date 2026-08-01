import { expect, type Page } from '@playwright/test';

/** Resolve once the app is fully live: `window.goofi` is published (AppShell mounted) and the
 * node catalog has arrived over the control WS (`query.nodeTypes()` non-empty ⇒ `hello` landed).
 * This is the readiness gate every spec waits on before driving the façade. */
export async function waitForApp(page: Page): Promise<void> {
	await page.waitForFunction(
		() => {
			const g = (window as any).goofi;
			return !!g && (g.query.nodeTypes()?.length ?? 0) > 0;
		},
		undefined,
		{ timeout: 20_000 }
	);
	await expectPristineWorkspace(page);
}

/**
 * The hermeticity backstop, and it lives here because the readiness gate is the one line every
 * product spec runs before it touches anything.
 *
 * One backend serves the whole run (`fullyParallel: false`, `workers: 1`, no globalSetup) and
 * AppShell pushes the layout into the RUNNING PATCH on a 400ms debounce that can outlive the page.
 * So a spec that splits a panel or adds a tab and leaves early persists that arrangement, `hello`
 * echoes it, and every later spec boots into it. This suite has been bitten by it four times, and
 * every time the red landed on an innocent file measuring geometry — never on the file that leaked.
 *
 * Asserting it at ENTRY is what changes that: the failure names the cause, in the first spec after
 * the leak, instead of surfacing as a wrong bounding box in another project. Two counts against an
 * already-loaded page; nothing to do in the normal case. A spec that means to leave the arrangement
 * changed does not exist — the rule is that it hands the workspace back in a `finally`.
 */
export async function expectPristineWorkspace(page: Page): Promise<void> {
	// The app is already live, so the default arrangement is on screen: a short budget keeps a leak
	// cheap to report instead of paying the 10s default on every spec after the one that leaked.
	const settled = { timeout: 2_000 };
	await expect(
		page.getByTestId('workspace-tabs').locator('.tab'),
		'a previous spec left an extra workspace tab behind — hand it back in a `finally`'
	).toHaveCount(1, settled);
	await expect(
		page.locator('.panel'),
		'a previous spec left a split panel behind — hand it back in a `finally`'
	).toHaveCount(1, settled);
}

/** Hand back a tab the spec added: close the last one and settle past AppShell's 400ms `set_layout`
 * debounce, so the arrangement never reaches the running patch. The `finally` half of the contract
 * `expectPristineWorkspace` enforces at the other end. */
export async function closeAddedTab(page: Page): Promise<void> {
	const tabs = page.getByTestId('workspace-tabs');
	await tabs.getByRole('button', { name: 'Close tab' }).last().click();
	await expect(tabs.locator('.tab'), 'the workspace is back to one tab').toHaveCount(1);
	await page.waitForTimeout(700);
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
 * Put the workspace back, through the split panel's own ✕. `ws.split` re-arms AppShell's 400ms
 * `set_layout` debounce, which writes into the RUNNING PATCH — one backend for the whole run — so a
 * spec that splits and leaves early persists a 2-panel workspace that every later spec boots into.
 * It passes alone and depends on nothing but screenshot latency, which is why it must be a `finally`.
 */
export async function closeSplit(page: Page): Promise<void> {
	await page.getByTestId('panel-header').nth(1).getByRole('button', { name: 'Close panel' }).click();
	await expect(page.locator('.panel'), 'the workspace is back to one panel').toHaveCount(1);
	await page.waitForTimeout(700); // past AppShell's 400ms set_layout debounce
}
