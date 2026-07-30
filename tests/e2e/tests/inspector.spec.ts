import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, nodeParams } from '../lib/goofi';

// Characterization e2e for the rebuilt inspector (spec §7, N-Task 5): drive the REAL rendered
// InspectorOverlay — the slide-in ParamForm — against a live Oscillator node, complementing the
// store-level graph.spec.ts. Each case commits through a rendered control and reads the round-trip
// back through the doc (`nodeParams` / `query.node`), so it pins the newly-wired ParamForm, not a
// synthetic fixture. Runs under the `default` (fine-pointer) project.

/** The single-selection slide-in inspector. */
function panel(page: Page) {
	return page.getByTestId('auto-side-panel');
}

/** A node's live name via the read façade. */
function nodeName(page: Page, uid: string): Promise<string | undefined> {
	return page.evaluate((u) => (window as any).goofi.query.node(u)?.name, uid);
}

/** Boot, add an Oscillator, select it so the per-editor inspector slides in, and return its uid. */
async function addAndSelect(page: Page): Promise<string> {
	await page.goto('/');
	await waitForApp(page);
	const uid = await addNode(page, 'Oscillator', 'inputs');
	await waitForNode(page, uid);
	// Selecting exactly one node opens the editor's inspector overlay (enabled by default).
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
	await expect(panel(page), 'the inspector slides in for a single selection').toHaveClass(/open/);
	// The backend persists across specs, so the auto-assigned display name is not fixed; assert the
	// header reflects THIS node's actual name (proving the overlay is bound to the selection).
	const name = await nodeName(page, uid);
	await expect(panel(page).getByTestId('node-name')).toHaveText(name!);
	return uid;
}

test.describe('Inspector (real node)', () => {
	test('commits a param through the rendered control and it round-trips through the doc', async ({
		page
	}) => {
		const uid = await addAndSelect(page);
		// amplitude lives in the default (`oscillator`) group — edit its NumberInput and commit on Enter.
		const number = panel(page).getByTestId('param-field-amplitude').getByTestId('param-number');
		await number.fill('0.42');
		await number.press('Enter');
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.value)
			.toBeCloseTo(0.42, 5);
	});

	test('toggles fx on a param and expression_enabled flips', async ({ page }) => {
		const uid = await addAndSelect(page);
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.expression_enabled)
			.toBe(false);
		await panel(page).getByTestId('param-field-amplitude').getByTestId('param-fx-toggle').click();
		await expect
			.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.expression_enabled)
			.toBe(true);
	});

	test('inline-renames the node from the header', async ({ page }) => {
		const uid = await addAndSelect(page);
		await panel(page).getByTestId('node-name').click();
		const input = panel(page).getByTestId('node-name-input');
		await input.fill('my_osc');
		await input.press('Enter');
		await expect.poll(() => nodeName(page, uid)).toBe('my_osc');
	});

	// The commit/cancel dance against the REAL store (the node's name IS store-bound, unlike the
	// gallery's offline sample): Escape must NOT commit. cancelRename nulls the captured uid so the
	// unmount-blur it triggers is a no-op — if commitRename lost that guard, the blur would wrongly
	// rename the node to the discarded draft.
	test('Escape during a header rename cancels — the store name stays unchanged', async ({ page }) => {
		const uid = await addAndSelect(page);
		const original = await nodeName(page, uid);
		await panel(page).getByTestId('node-name').click();
		const input = panel(page).getByTestId('node-name-input');
		await input.fill('DRAFT_DISCARD_ME'); // a draft DISTINCT from the current name
		await input.press('Escape');
		// The editor closes and the store name is unchanged — the unmount-blur committed nothing.
		await expect(panel(page).getByTestId('node-name-input'), 'Escape closes the editor').toHaveCount(0);
		await expect
			.poll(() => nodeName(page, uid), { message: 'a cancel must never commit the draft' })
			.toBe(original);
	});

	test('opens the docs disclosure to reveal the node docstring', async ({ page }) => {
		await addAndSelect(page);
		// The Disclosure keeps its body out of the DOM until opened.
		await expect(panel(page).getByTestId('docstring')).toHaveCount(0);
		await panel(page).getByTestId('docs-toggle').click();
		await expect(panel(page).getByTestId('docstring')).toBeVisible();
	});

	// --- state-coupling seam (audit findings #1/#2/#11) ------------------------------------------
	// The ParamForm rows must be keyed by node identity, and the fx editor's global standdown must be
	// ref-counted + released when a field leaves expression mode. These two cases drive the REAL two-node
	// switch that the per-instance ParamField state used to survive.

	const modalOpen = (page: Page): Promise<boolean> =>
		page.evaluate(() => (window as any).goofi.query.modalOpen());
	const historyLen = (page: Page): Promise<number> =>
		page.evaluate(() => (window as any).goofi.query.historyLength());

	test('a node switch tears down an open fx editor and frees global undo/redo', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// Two oscillators: A gets an open fx editor; B keeps fx OFF on the same-named param.
		const a = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, a);
		const b = await addNode(page, 'Oscillator', 'inputs', [280, 0]);
		await waitForNode(page, b);

		await page.evaluate((u) => (window as any).goofi.commands.select([u]), a);
		await expect(panel(page)).toHaveClass(/open/);

		// Enable fx on amplitude and grow the in-panel multi-line editor — it owns the keyboard, so the
		// global undo/redo stands down.
		const ampA = panel(page).getByTestId('param-field-amplitude');
		await ampA.getByTestId('param-fx-toggle').click();
		await ampA.getByTestId('param-expr-expand').click();
		await expect(ampA.getByTestId('param-expr-multiline')).toBeVisible();
		await expect.poll(() => modalOpen(page)).toBe(true);

		// Switch selection to B (fx OFF on its amplitude). The field must tear down: no lingering editor,
		// and the standdown must lift (before the fix `modalOpen` stayed stuck true → undo went dead).
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), b);
		await expect(panel(page).getByTestId('param-expr-multiline')).toHaveCount(0);
		await expect.poll(() => modalOpen(page)).toBe(false);

		// End-to-end proof the standdown lifted: a real Ctrl-Z now performs an undo through the app's
		// keybinding path (AppShell → undoKeyAction(ui().modalOpen)), which was dead while modalOpen stuck.
		const before = await historyLen(page);
		await page.keyboard.press('Control+z');
		await expect.poll(() => historyLen(page)).toBeLessThan(before);
	});

	// M-audit CRITICAL. `openEditor`/`closeEditor` READ the `$state` Set they then reassign, so every
	// caller's `$effect` becomes a dependency of what it writes. Svelte runs an effect's teardown BEFORE
	// its body on re-run, so each re-run writes twice — harmless while there is ONE registrant (the
	// `has(id)` guard short-circuits), but with TWO the effects invalidate each other forever until the
	// flush guard throws `effect_update_depth_exceeded` and leaves the batch dead: every later state
	// change silently stops applying. Two ordinary clicks reach it. The suite never caught it because
	// `ui.svelte.test.ts` calls the mutators outside any effect and every e2e opened only one editor.
	test('two concurrently open fx editors do not kill the reactive graph', async ({ page }) => {
		const pageErrors: string[] = [];
		page.on('pageerror', (e) => pageErrors.push(e.message));

		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, uid);
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		await expect(panel(page)).toHaveClass(/open/);

		// Two registrants at once — one multi-line fx editor on each of two params of the SAME node.
		// ParamForm renders one ParamField per param with no single-open coordination, and each mints a
		// globally unique `$props.id()`, so the idempotence guard cannot absorb the second.
		for (const name of ['amplitude', 'frequency']) {
			const field = panel(page).getByTestId(`param-field-${name}`);
			await field.getByTestId('param-fx-toggle').click();
			await field.getByTestId('param-expr-expand').click();
			await expect(field.getByTestId('param-expr-multiline')).toBeVisible();
		}
		expect(pageErrors).toEqual([]);

		// The real proof is that state still FLOWS. Collapse one editor: the other still holds the
		// standdown, so modalOpen must stay true — and it can only stay/settle correctly if the batch is
		// alive. With the batch dead these polls never observe another change.
		await panel(page)
			.getByTestId('param-field-amplitude')
			.getByTestId('param-expr-collapse')
			.click();
		await expect.poll(() => modalOpen(page)).toBe(true);

		await panel(page)
			.getByTestId('param-field-frequency')
			.getByTestId('param-expr-collapse')
			.click();
		await expect.poll(() => modalOpen(page)).toBe(false);
		expect(pageErrors).toEqual([]);
	});

	// The second reachable pairing: an open fx editor plus the file browser, which registers the same
	// standdown. The TopBar buttons are NOT gated on modalOpen (only the Ctrl+S/Ctrl+O chords are), so
	// the Parameters panel and its live effect stay mounted underneath.
	test('opening the file browser over an fx editor does not kill the reactive graph', async ({
		page
	}) => {
		const pageErrors: string[] = [];
		page.on('pageerror', (e) => pageErrors.push(e.message));

		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, uid);
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
		await expect(panel(page)).toHaveClass(/open/);

		const amp = panel(page).getByTestId('param-field-amplitude');
		await amp.getByTestId('param-fx-toggle').click();
		await amp.getByTestId('param-expr-expand').click();
		await expect(amp.getByTestId('param-expr-multiline')).toBeVisible();

		await page.getByTestId('topbar-load').click();
		await expect(page.getByTestId('fs-browser')).toBeVisible();
		expect(pageErrors).toEqual([]);

		// Dismiss the browser; the fx editor still holds the standdown, which requires a live batch.
		// Wait for the dialog to be genuinely MODAL, not merely visible: `Dialog` calls `showModal()`
		// from an `$effect`, so for one frame the element is in the DOM (and `toBeVisible` passes) while
		// focus is still in the fx textarea — where Escape hits ParamField's own handler and collapses
		// the editor instead of closing the browser. `dialog[open]` is set by `showModal()` itself, so
		// it is the exact signal that focus has moved in.
		await expect(page.locator('dialog[open]')).toHaveCount(1);
		// Opening a modal must not collapse the editor underneath it — pinned here so a failure below
		// is attributable to the dismissal rather than to the open.
		await expect(amp.getByTestId('param-expr-multiline')).toBeVisible();

		// Dismiss by the ✕, deliberately NOT by Escape. Escape is ambiguous while both surfaces are up:
		// `Dialog` calls `showModal()` from an `$effect`, and on close it restores focus to the fx
		// textarea, whose own keydown handler treats Escape as "collapse the editor". That made this
		// test flake ~1/20 on a state it does not exist to check. Escape-dismissal of the browser is
		// already covered by `fs-browser.spec.ts`; this test is about the standdown ref-count.
		await page.getByTestId('fs-browser').getByRole('button', { name: 'Close' }).click();
		await expect(page.getByTestId('fs-browser')).toHaveCount(0);
		await expect.poll(() => modalOpen(page)).toBe(true);
		expect(pageErrors).toEqual([]);
	});

	test('a node switch does not leak an open fx buffer onto another node’s param', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const a = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, a);
		const b = await addNode(page, 'Oscillator', 'inputs', [280, 0]);
		await waitForNode(page, b);

		// Give B its OWN fx expression on amplitude, so B's row also renders the expression control.
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), b);
		await panel(page).getByTestId('param-field-amplitude').getByTestId('param-fx-toggle').click();
		await expect
			.poll(async () => (await nodeParams(page, b))?.oscillator?.amplitude?.expression_enabled)
			.toBe(true);
		const bExpr: string = (await nodeParams(page, b))?.oscillator?.amplitude?.expression ?? '';

		// On A: enable fx, open the multi-line editor, type a DISTINCTIVE expression — but do NOT apply.
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), a);
		const ampA = panel(page).getByTestId('param-field-amplitude');
		await ampA.getByTestId('param-fx-toggle').click();
		await ampA.getByTestId('param-expr-expand').click();
		await ampA.getByTestId('param-expr-multiline').fill('nd("LEAK_FROM_A").out.data');

		// Switch to B (whose amplitude also has fx on). Before the fix the SAME field survived, keeping A's
		// buffer in an open textarea now wired to B — one apply-click from silently corrupting B.
		await page.evaluate((u) => (window as any).goofi.commands.select([u]), b);
		const ampB = panel(page).getByTestId('param-field-amplitude');
		// The field remounted for B: no editor carries A's buffer; B shows its own inline expression.
		await expect(ampB.getByTestId('param-expr-multiline')).toHaveCount(0);
		// `toHaveText`, not `toHaveValue`: since X the field is a CodeMirror document, not an <input>.
		await expect(ampB.getByTestId('param-expr-input')).toHaveText(bExpr);
		// And B's committed expression was never overwritten with A's.
		expect((await nodeParams(page, b))?.oscillator?.amplitude?.expression).not.toContain('LEAK_FROM_A');
	});
});
