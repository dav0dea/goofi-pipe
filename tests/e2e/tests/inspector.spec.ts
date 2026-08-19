// The parameter inspector: editing a value, binding an expression to it, and how the form
// reflows against its host.

import { test, expect, type Page, type Locator } from '@playwright/test';
import { waitForApp, appReady, closeSplit, splitRight } from '../lib/app';
import { settledBox } from '../lib/geometry';
import {
	addErroringNode,
	addGlobal,
	addNode,
	canUndo,
	nodeParams,
	selectNode,
	undo,
	updateParam,
	waitForNoNode,
	waitForNode
} from '../lib/goofi';
import {
	dropNode as drop,
	grip,
	openInspector as addAndSelect,
	pane,
	paneAxis,
	paneBound,
	rootRem as rem,
	settledGrabber,
	type Grabber
} from '../lib/inspector';

test.describe('editing parameters', () => {
	// Characterization e2e for the rebuilt inspector (spec §7, N-Task 5): drive the REAL rendered
	// InspectorOverlay — the slide-in ParamForm — against a live Oscillator node, complementing the
	// store-level graph.spec.ts. Each case commits through a rendered control and reads the round-trip
	// back through the doc (`nodeParams` / `query.node`), so it pins the newly-wired ParamForm, not a
	// synthetic fixture. Runs under the `default` (fine-pointer) project.

	/** The single-selection slide-in inspector. */
	function panel(page: Page) {
		return page.getByTestId('auto-side-panel');
	}

	// The shared-backend half of the hermeticity contract (`expectPristineWorkspace`): every node on
	// the backend at teardown is THIS file's creation — the guard proved the graph empty at entry — so
	// one remove-all here beats nine per-test `finally` blocks, and it still runs when a test fails.
	test.afterEach(async ({ page }) => {
		const uids: string[] = await page.evaluate(() =>
			(window as any).goofi.query.graph().nodes.map((n: { uid: string }) => n.uid)
		);
		if (uids.length === 0) return;
		await page.evaluate((us) => (window as any).goofi.commands.removeNodes(us), uids);
		await expect
			.poll(() => page.evaluate(() => (window as any).goofi.query.graph().nodes.length))
			.toBe(0);
	});

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
		await selectNode(page, uid);
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

			await selectNode(page, a);
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
			await selectNode(page, b);
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
			await selectNode(page, uid);
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
			await selectNode(page, uid);
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
			await selectNode(page, b);
			await panel(page).getByTestId('param-field-amplitude').getByTestId('param-fx-toggle').click();
			await expect
				.poll(async () => (await nodeParams(page, b))?.oscillator?.amplitude?.expression_enabled)
				.toBe(true);
			const bExpr: string = (await nodeParams(page, b))?.oscillator?.amplitude?.expression ?? '';

			// On A: enable fx, open the multi-line editor, type a DISTINCTIVE expression — but do NOT apply.
			await selectNode(page, a);
			const ampA = panel(page).getByTestId('param-field-amplitude');
			await ampA.getByTestId('param-fx-toggle').click();
			await ampA.getByTestId('param-expr-expand').click();
			await ampA.getByTestId('param-expr-multiline').fill('nd("LEAK_FROM_A").out.data');

			// Switch to B (whose amplitude also has fx on). Before the fix the SAME field survived, keeping A's
			// buffer in an open textarea now wired to B — one apply-click from silently corrupting B.
			await selectNode(page, b);
			const ampB = panel(page).getByTestId('param-field-amplitude');
			// The field remounted for B: no editor carries A's buffer; B shows its own inline expression.
			await expect(ampB.getByTestId('param-expr-multiline')).toHaveCount(0);
			// `toHaveText`, not `toHaveValue`: since X the field is a CodeMirror document, not an <input>.
			await expect(ampB.getByTestId('param-expr-input')).toHaveText(bExpr);
			// And B's committed expression was never overwritten with A's.
			expect((await nodeParams(page, b))?.oscillator?.amplitude?.expression).not.toContain('LEAK_FROM_A');
		});

		// A node's traceback is machine output — the same text the ErrorPanel shows in mono — so the
		// inspector's copy reads in mono too (spec D-T3). It is a bare <pre>, i.e. an element whose only
		// family came from app.css's `code, pre, kbd { font: inherit }` reset over a mono body: the
		// two-face flip turned that inheritance sans without touching this file, which is exactly the
		// class of regression this pins. The error is a real per-tick one: a ticking node whose required
		// input slot is empty (see `addErroringNode` for the whole mechanism).
		test('a node error renders its traceback in mono (D-T3)', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addErroringNode(page);
			await selectNode(page, uid);
			const pre = panel(page).getByTestId('inspector-error').locator('pre');
			await expect(pre, 'the real per-tick error reached the inspector').toBeVisible();
			expect(
				await pre.evaluate((el) => getComputedStyle(el).fontFamily.split(',')[0].replace(/["']/g, '')),
				'the traceback'
			).toBe('JetBrains Mono');
		});
	});
});

test.describe('dismissing the inspector', () => {
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
		await selectNode(page, uid);
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
		await selectNode(page, uid);
		await expect(pane(page), 're-selecting revives the pane').toHaveClass(/open/);

		// Dismiss again, then select a DIFFERENT node directly → the pane returns for it.
		await pane(page).getByTestId('inspector-close').click();
		await expectParked(page);
		// Clear of the first node: with a real click the two cards must not overlap, or the press
		// lands on whichever one is on top.
		const other = await addNode(page, 'Buffer', 'inputs', [40, 300]);
		await waitForNode(page, other);
		await selectNode(page, other);
		await expect(pane(page), 'a different node revives the pane').toHaveClass(/open/);

		// The ◧, by contrast, IS the off-switch: turn the inspector off with it and selection
		// changes stay silent.
		await pane(page).getByTestId('inspector-close').click();
		await expectParked(page);
		await page.getByTestId('inspector-toggle').click(); // shows (clears the dismissal)
		await expect(pane(page)).toHaveClass(/open/);
		await page.evaluate(() => (window as any).goofi.commands.select([]));
		await page.getByTestId('inspector-toggle').click(); // parked + visible → this press disables
		await selectNode(page, uid);
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
});

test.describe('the inspector reflows with its host', () => {
	/**
	 * The orientation-aware inspector, on the FINE pointer — because the anchor rule contains no device
	 * class at all (D-I2). It asks its host panel whether it is taller than it is wide, through
	 * `@container (orientation: portrait)`, and a desktop panel can answer either way.
	 *
	 * That is the point of this file being a `default`-project spec: everything here runs on a mouse,
	 * at a landscape desktop window, and one of the cases still gets the bottom sheet. `touch-sheet` and
	 * `touch-reflow` cover the phone geometries, and `touch-modality` runs the coarse half of the very
	 * assertions below — off `lib/inspector.ts`, which both files read the pane through, so the mouse
	 * and the finger are measured with one ruler.
	 *
	 * Sizes are measured against the HOST or against what the pane itself publishes, never against a
	 * literal — except where the literal IS the claim (§4.3: the desktop resting width does not move
	 * off 420px). In particular no test here restates the small-screen guard's PERCENTAGE: which of the
	 * resting size's two halves bound is asserted by measurement, so D-I6 can retune that number without
	 * this file having an opinion about it.
	 */

	test('the desktop resting width does not move off 420px (§4.3)', async ({ page }) => {
		// 1600 wide, so the ceiling resolves to its REM half: the root clamp saturates at 14px anywhere
		// near this size, so 30rem is exactly 420 — the width the pane has always rested at. On a
		// narrower host the small-screen guard binds instead, which is the next case.
		await page.setViewportSize({ width: 1600, height: 900 });
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		expect(await rem(page), 'the root clamp is saturated at this size').toBeCloseTo(14, 5);
		const uid = await addAndSelect(page);
		try {
			const p = await settledBox(pane(page));
			expect(p.width, 'the resting pane is 30rem = 420px').toBeCloseTo(420, 0);
			// …and 30rem is genuinely the binding half here, MEASURED rather than arithmetic: a wider
			// host cannot make the pane wider. Stated this way it names no percentage, so it holds for
			// whatever the small-screen guard is set to — which is the number D-I6 leaves live.
			await page.setViewportSize({ width: 1900, height: 900 });
			const wider = await settledBox(pane(page));
			expect(wider.width, 'and a wider host cannot make it wider').toBeCloseTo(420, 0);
		} finally {
			await drop(page, uid);
		}
	});

	/**
	 * The other half of D-I6's resting size, and the half the small-screen guard exists for: on a host
	 * too narrow for the 30rem comfort cap, it is the HOST that decides. Before D-I6 the pane sat at a
	 * flat 420px here and its only host clamp was `100% - --hit`, which reserved 28px of canvas out of
	 * 1272.
	 *
	 * No fraction is restated: what is asserted is which of the two halves bound, and that the one that
	 * bound still left the pane room. That second assertion is the whole of the floor/ceiling fix — a
	 * host-relative size is free to resolve BELOW the floor it is measured against, which is exactly
	 * what a landscape phone got (256px under a 260px floor), and stating the floor as the SAME
	 * `clamp()`'s lower bound is what stops it. A guard tight enough to cross the floor on a narrow
	 * desktop fails right here.
	 */
	test('a narrower desktop editor sizes the pane against its HOST, and still leaves it room', async ({
		page
	}) => {
		// 800 wide: narrow enough that the small-screen guard is the tighter half at this root size.
		await page.setViewportSize({ width: 800, height: 720 });
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			const p = await settledBox(pane(page));
			expect(p.width, 'the 30rem comfort cap is not what bound it — the host did').toBeLessThan(
				30 * (await rem(page))
			);
			expect(
				p.width,
				'…and the host-relative size did not land on or under the floor beside it'
			).toBeGreaterThan(await paneBound(page, 'floor'));
		} finally {
			await drop(page, uid);
		}
	});

	/**
	 * The consequence the spec flagged for the user rather than choosing silently (§6), pinned so it
	 * reads as deliberate.
	 *
	 * The WINDOW is landscape here — `matchMedia` says so — and the pane is a bottom sheet anyway,
	 * because the panel it lives in is a narrow tall column. That is the whole argument for D-I2 over
	 * `@media (orientation)`: this docked editor has exactly the phone's problem (a right-hand pane
	 * would eat the canvas), and a viewport query would answer "landscape" and leave it broken. It is
	 * also the one line that would be edited to take it back — see `InspectorOverlay.svelte`'s
	 * `@media all` prelude.
	 */
	test('a narrow, tall DOCKED editor gets the sheet, in a landscape window (spec §6)', async ({
		page
	}) => {
		await page.setViewportSize({ width: 1280, height: 900 });
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		expect(
			await page.evaluate(() => matchMedia('(orientation: landscape)').matches),
			'the WINDOW is landscape — the sheet below is the panel’s shape, not the screen’s'
		).toBe(true);
		await splitRight(page);
		let uid = '';
		try {
			// Both halves are node editors and each mounts its own inspector, so everything below is
			// scoped to the LEFT one — and its selection is addressed by panel id rather than through the
			// active editor, which the split just moved to the new panel.
			const left = page.locator('.panel').first();
			const panelId: string = await page.evaluate(
				() => (window as any).goofi.query.panels()[0].panelId
			);
			uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			// Click the card IN the left panel: which panel a selection lands in is what this
			// scenario is about, and both panels render the same node.
			await left.locator(`.svelte-flow__node[data-id="${uid}"] .header`).click();
			await page.waitForFunction(
				([u, id]) => ((window as any).goofi.query.selection(id).nodes as string[]).includes(u),
				[uid, panelId] as const
			);
			const sheet = left.getByTestId('auto-side-panel');
			await expect(sheet, 'the left editor inspects its selection').toHaveClass(/open/);

			const host = (await left.locator('.editor-panel').boundingBox())!;
			expect(host.height, 'the split left the editor taller than it is wide').toBeGreaterThan(
				host.width
			);

			const p = await settledBox(sheet);
			expect(p.width, 'the sheet spans its host').toBeCloseTo(host.width, 0);
			expect(
				host.y + host.height - (p.y + p.height),
				'and is flush with the bottom it slid from'
			).toBeLessThan(2);
			expect(p.height, 'at the 60% D-I6 allows').toBeCloseTo(host.height * 0.6, 0);
		} finally {
			if (uid) await drop(page, uid);
			await closeSplit(page);
		}
	});

	/**
	 * D-I3/D-I4: the edge drag is not a touch affordance. It is THE resize, identical on both inputs,
	 * and this is the mouse half — driven with `page.mouse`, whose `pointerType` is `mouse`, i.e. the
	 * input every coarse door in the app is closed to.
	 */
	test('an edge drag resizes the pane with a MOUSE, and the size outlives the reload', async ({
		page
	}) => {
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			const before = await settledBox(pane(page));
			const g = (await grip(page).boundingBox())!;
			const y = g.y + g.height / 2;
			// Rightward, i.e. INTO the pane, which shrinks it — the same direction of travel that shrinks
			// the sheet when it is pushed down. Away from the pane the ceiling would bind and the
			// measurement would be of `max-width`, not of the drag.
			await page.mouse.move(g.x + g.width / 2, y);
			await page.mouse.down();
			await page.mouse.move(g.x + g.width / 2 + 60, y, { steps: 8 });
			await page.mouse.up();

			const after = await settledBox(pane(page));
			expect(after.width, 'the pane shrank by exactly the drag').toBeCloseTo(before.width - 60, 0);
			expect(
				await page.evaluate(() => localStorage.getItem('goofi.panelWidth')),
				'and the RENDERED size is what was stored, so a reload agrees with the screen'
			).toBe(String(Math.round(after.width)));

			await page.reload();
			// Readiness only: this spec's own node is legitimately still on the shared backend, which
			// is exactly the state `waitForApp`'s hermeticity backstop exists to reject.
			await appReady(page);
			await selectNode(page, uid);
			await expect(pane(page)).toHaveClass(/open/);
			const restored = await settledBox(pane(page));
			expect(restored.width, 'and it comes back at that width').toBeCloseTo(after.width, 0);
		} finally {
			await drop(page, uid);
		}
	});

	/**
	 * The mouse half of THE GESTURE IS UNIFORM: the drag `touch-modality.spec.ts` carries past the floor
	 * with a finger, carried just as far with a MOUSE. Both resize to the floor and leave the pane open,
	 * which is the whole claim — there is no pointer type for which this gesture means something else.
	 */
	test('a drag carried far past the floor clamps there, on a mouse too (D-I4)', async ({ page }) => {
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			const before = await settledBox(pane(page));
			// The floor is the STYLESHEET's — the lower bound of the pane's own `clamp()`, asked of CSS
			// rather than a `260` this file would hold its own copy of.
			const floor = await paneBound(page, 'floor');
			const g = (await grip(page).boundingBox())!;
			const y = g.y + g.height / 2;
			await page.mouse.move(g.x + g.width / 2, y);
			await page.mouse.down();
			// Past the floor by the pane's whole width again, so nothing about this ends AT the bound.
			await page.mouse.move(g.x + g.width / 2 + before.width + 100, y, { steps: 10 });
			await page.mouse.up();

			await expect(pane(page), 'the pane is still there').toHaveClass(/open/);
			const after = await settledBox(pane(page));
			expect(after.width, 'clamped at the floor, not closed').toBeCloseTo(floor, 0);
		} finally {
			await drop(page, uid);
		}
	});

	/**
	 * D-I9, and the rule it is one instance of: ORIENTATION decides the anchored AXIS; INPUT MODALITY
	 * decides only the resting AFFORDANCE. The two are independent — so under one modality the grabber
	 * must be the SAME in either anchor, and this reads it in both and compares them against each other
	 * rather than against a number either could drift away from alone.
	 *
	 * It was not. The portrait branch declared a resting pill unconditionally, which is an affordance
	 * chosen by orientation: a phone got a chunky pill standing up and a thin line lying down (the same
	 * finger, two affordances), and a narrow docked desktop column got the touch grabber under a mouse.
	 * The pill belongs to the coarse block — `touch-modality.spec.ts` proves it there, in BOTH anchors —
	 * and what portrait carries is geometry alone.
	 *
	 * What a fine pointer gets is therefore the transparent-until-hover seam in either anchor. That is
	 * not an affordance living solely behind `:hover` in CLAUDE.md's sense: hover is the door this
	 * modality HAS, which is exactly why the door a finger needs is gated on the finger and not on
	 * which way the phone is held.
	 */
	test('one modality, one grabber — the fine seam is identical in either anchor', async ({ page }) => {
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const seen: Partial<Record<'portrait' | 'landscape', Grabber>> = {};
		for (const [name, size, axis] of [
			['portrait', { width: 600, height: 1000 }, 'y'],
			['landscape', { width: 1280, height: 800 }, 'x']
		] as const) {
			await page.setViewportSize(size);
			const uid = await addAndSelect(page);
			try {
				// The pointer is parked off the pane — no hover anywhere near the grip.
				await page.mouse.move(5, 5);
				await expect
					.poll(() => paneAxis(page), { message: `${name} anchors as its host panel is shaped` })
					.toBe(axis);
				seen[name] = await settledGrabber(page);
				// …and the seam is not merely invisible, it is hover-REVEALED, in this anchor too. That
				// door has to exist wherever the transparent resting state does, or the affordance really
				// would be nowhere on a fine pointer.
				await grip(page).hover();
				await expect
					.poll(() => settledGrabber(page).then((g) => g.painted), {
						message: `hovering the ${name} seam lights it`
					})
					.toBe(true);
			} finally {
				await drop(page, uid);
			}
		}
		expect(seen.portrait, 'one modality, one grabber — whichever axis it is on').toEqual(
			seen.landscape
		);
		expect(
			seen.portrait!.painted,
			'a fine pointer rests on the transparent seam, having hover to find it with'
		).toBe(false);
		expect(seen.portrait!.length, 'a hairline down the whole seam, never a pill').toBe('seam');
		expect(seen.portrait!.rounded, 'and uncapped — the pill is the coarse pointer’s').toBe(false);
	});

	test('the ✕ dismisses the pane in either anchor', async ({ page }) => {
		// `inspector-dismiss.spec.ts` proves the ✕ on the right-hand pane; this is the same door on the
		// SHEET, and D-I4 is why it must exist there too — it is the ONLY way out, in either anchor.
		await page.setViewportSize({ width: 600, height: 1000 });
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const uid = await addAndSelect(page);
		try {
			await expect
				.poll(() =>
					pane(page).evaluate((el) => getComputedStyle(el).getPropertyValue('--pane-axis').trim())
				)
				.toBe('y');
			await pane(page).getByTestId('inspector-close').click();
			await expect(pane(page), 'the sheet closes').not.toHaveClass(/open/);
			await expect(pane(page), 'and finishes parked off-screen').toHaveCSS('visibility', 'hidden');
		} finally {
			await drop(page, uid);
		}
	});
});

test.describe('the expression editor', () => {
	/**
	 * The param expression editor (sub-project X), driven in the REAL app against a live graph — because
	 * every interesting thing about it is a fact about the running document: which classes the
	 * highlighter actually emitted, which sources answered at the cursor, whether one popup or two
	 * appeared, and whether a keystroke reached the backend.
	 *
	 * `lib/inspector/expr/*.test.ts` owns the pure half (what a cursor position MEANS, where a
	 * diagnostic anchors, that a newline is refused). What cannot be unit-tested here is a mounted
	 * Svelte component, which is the whole reason this file exists.
	 *
	 * Runs under the `default` (fine-pointer) project; `touch-expr.spec.ts` is the coarse half.
	 */

	/** The single-selection slide-in inspector. */
	const pane = (page: Page) => page.getByTestId('auto-side-panel');
	/** The one completion popup. Its COUNT is load-bearing — see the merge test. */
	const popup = (page: Page) => page.locator('.cm-tooltip-autocomplete');
	/** The visible options, label + detail per row. */
	const options = async (page: Page): Promise<string[]> =>
		(await popup(page).count()) ? popup(page).locator('li').allInnerTexts() : [];

	/** Boot, add an Oscillator, select it, switch its `amplitude` into fx mode, and return the editor. */
	async function fxEditor(page: Page, uid: string): Promise<Locator> {
		await selectNode(page, uid);
		await expect(pane(page)).toHaveClass(/open/);
		const field = pane(page).getByTestId('param-field-amplitude');
		await field.getByTestId('param-fx-toggle').click();
		const editor = field.getByTestId('param-expr-input');
		await expect(editor, 'the fx editor takes over the control region').toBeVisible();
		return editor;
	}

	/** Replace the whole document with `src` by typing it, then settle past the completion debounce. */
	async function retype(page: Page, editor: Locator, src: string): Promise<void> {
		await editor.click();
		await page.keyboard.press('Control+a');
		await page.keyboard.press('Delete');
		await page.keyboard.type(src, { delay: 10 });
		await page.waitForTimeout(300);
	}

	test.describe('Param expression editor', () => {
		/* D-X5 / success criterion 6, measured on the wire rather than argued from the import graph: no
		   bundle the app fetches at boot contains the editor, and the fx toggle is what fetches it.
		   Keyed on a string only the CodeMirror chunk can contain, so it cannot pass by naming a hash. */
		test('the editor chunk is absent from every boot bundle and arrives on the first fx render', async ({
			page
		}) => {
			const MARK = 'cm-tooltip-autocomplete';
			const carrying: string[] = [];
			page.on('response', async (r) => {
				if (!/_app\/immutable\/.*\.js$/.test(r.url())) return;
				try {
					if ((await r.text()).includes(MARK)) carrying.push(r.url());
				} catch {
					/* a navigation can cancel a body read; a cancelled response was never applied anyway */
				}
			});
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			expect(carrying, 'first paint pays nothing for the editor').toEqual([]);
			try {
				await fxEditor(page, uid);
				await expect
					.poll(() => carrying.length, { message: 'the fx render fetches the editor chunk' })
					.toBe(1);
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});

		/* D-X2 / criteria 1 and 2. The generated highlight class names are hashes, so this asserts what
		   the user actually sees: the COMPUTED ink of each token against the `--syn-*` token app.css names
		   for that role, resolved through a probe element in the page so the comparison is in the
		   browser's own colour space. */
		test('paints each Python role in its own --syn-* ink, and the field never grows', async ({
			page
		}) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			try {
				const editor = await fxEditor(page, uid);
				await retype(page, editor, "nd('oscillator0').out  # note");
				const seen = await editor.evaluate((el) => {
					const probe = document.createElement('span');
					el.appendChild(probe);
					const ink = (token: string): string => {
						probe.style.color = `var(${token})`;
						return getComputedStyle(probe).color;
					};
					const want = {
						function: ink('--syn-function'),
						string: ink('--syn-string'),
						operator: ink('--syn-operator'),
						name: ink('--syn-name'),
						comment: ink('--syn-comment')
					};
					probe.remove();
					const got: Record<string, string> = {};
					for (const s of el.querySelectorAll('span'))
						got[(s.textContent ?? '').trim()] = getComputedStyle(s).color;
					return { want, got };
				});
				expect(seen.got['nd'], 'the callee reads as a function').toBe(seen.want.function);
				expect(seen.got["'oscillator0'"], 'the literal reads as a string').toBe(seen.want.string);
				expect(seen.got['.'], 'the accessor reads as an operator').toBe(seen.want.operator);
				expect(seen.got['out'], 'the property reads as a name').toBe(seen.want.name);
				expect(seen.got['# note'], 'the comment reads as a comment').toBe(seen.want.comment);

				// D-X1: single-line mode must not grow. Long source scrolls sideways instead of wrapping.
				const short = (await editor.boundingBox())!.height;
				await retype(
					page,
					editor,
					"nd('oscillator0').out.data.mean() * 1000 + 42.5 / 3.14159 - 7 ** 2"
				);
				expect((await editor.boundingBox())!.height, 'the inline field is still one line').toBe(
					short
				);
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});

		/**
		 * THE test for the whole sub-project (criterion 2, second clause). The user asked for the goofi
		 * completions to be built INTO the standard Python autocomplete rather than bolted on, and
		 * `pythonLanguage.data.of({ autocomplete })` is how: CodeMirror collects every source registered as
		 * language data at the cursor and merges them into one ranked list.
		 *
		 * So the proof is two assertions that a bolted-on implementation could not both pass: there is
		 * exactly ONE popup in the document, and it holds our `nd` AND Python's own builtins. A separate
		 * popup, or a wrapper that pre-empted Python's sources, fails one or the other.
		 */
		test('goofi and Python completions arrive in ONE popup, ranked together', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			try {
				const editor = await fxEditor(page, uid);
				await retype(page, editor, 'n');
				await expect(popup(page), 'one popup, not one per source').toHaveCount(1);
				const labels = (await options(page)).map((t) => t.split('\n')[0]);
				expect(labels, 'ours: the evaluator’s injected scope').toContain('nd');
				expect(labels, 'ours: numpy is in scope too').toContain('np');
				expect(
					labels.filter((l) => ['None', 'NameError', 'next', 'not'].includes(l)).length,
					'Python’s own sources answered the same cursor'
				).toBeGreaterThan(0);
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});

		/* Criterion 2, first clause: the node names are the LIVE graph's, and the slots are THAT node's —
		   read from the string literal in the syntax tree, which is why `nd('psd0').` cannot answer with the
		   oscillator's slot. Accepting finishes the call (D-X10). */
		test('nd(…) offers the live graph’s nodes, and nd(x). that node’s own slots', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, osc);
			// A node whose output slot is NOT called `out`, so "that node's slots" is a real claim.
			const psd = await addNode(page, 'Psd', 'python', [360, 40]);
			await waitForNode(page, psd);
			try {
				const editor = await fxEditor(page, osc);
				await retype(page, editor, "nd('");
				const names = (await options(page)).map((t) => t.split('\n')[0]);
				expect(names, 'the live graph’s node names').toEqual(
					expect.arrayContaining(['oscillator0', 'psd0'])
				);

				await retype(page, editor, "nd('psd0').");
				expect(
					(await options(page)).map((t) => t.split('\n')[0]),
					'the Psd node’s own output slot, not the oscillator’s'
				).toEqual(['psd']);

				// Accept by click: the name completes AND the call is finished, so the expression is valid.
				await retype(page, editor, "nd('psd");
				await popup(page).locator('li').first().click();
				await expect(editor, 'accepting finishes the call').toHaveText("nd('psd0')");
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				for (const u of [osc, psd]) {
					await page.evaluate((x) => (window as any).goofi.commands.removeNode(x), u);
					await waitForNoNode(page, u).catch(() => {});
				}
			}
		});

		/* Criterion 3. `globals.` reads the patch's globals doc root (`default_ufreq` is always there, and
		   the added one proves it is live rather than a static list); `np.` is the curated surface (D-X9). */
		test('globals. offers the patch’s globals and np. the curated numpy surface', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			const G = 'expr_probe_gain';
			await addGlobal(page, G, 2.5, 'float');
			try {
				const editor = await fxEditor(page, uid);
				await retype(page, editor, 'globals.');
				const globals = (await options(page)).map((t) => t.split('\n')[0]);
				expect(globals, 'the system global the engine always defines').toContain('default_ufreq');
				expect(globals, 'and the one this spec just added — a live read, not a fixed list').toContain(G);

				await retype(page, editor, 'np.me');
				expect((await options(page)).map((t) => t.split('\n')[0])).toEqual(
					expect.arrayContaining(['mean', 'median'])
				);
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((n) => (window as any).goofi.commands.removeGlobal(n), G);
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});

		/* Criterion 4, on the gallery's fixed errored descriptor so it needs no live eval (the suite's
		   backend has no numpy, per the README). BOTH surfaces are asserted: the squiggle carries the
		   position, the row carries the text — a message that existed only behind the squiggle's hover
		   would be information behind hover, which the project forbids. */
		test('an errored expression draws an inline diagnostic AND shows the message in the row', async ({
			page
		}) => {
			await page.goto('/dev/inspector');
			const field = page.getByTestId('inspector-fx-error');
			const row = field.getByTestId('param-expr-error');
			await expect(row, 'the always-visible error row').toBeVisible();
			await expect(row).toContainText('NameError');
			await expect(
				field.getByTestId('param-expr-input').locator('.cm-lintRange-error'),
				'and the source itself is marked'
			).toHaveCount(1);
		});

		/* D-X6: a commit is an RPC plus a re-eval, so typing must not commit — only Enter (inline) does. */
		test('typing does not commit; Enter does, and it reaches the backend', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			try {
				const editor = await fxEditor(page, uid);
				const expr = async () =>
					(await nodeParams(page, uid))?.oscillator?.amplitude?.expression ?? '';
				const seeded = await expr();
				await retype(page, editor, '0.25 + 0.5');
				expect(await expr(), 'the keystrokes stayed local').toBe(seeded);
				await editor.press('Enter');
				await expect
					.poll(expr, { message: 'Enter commits the source to the backend' })
					.toBe('0.25 + 0.5');
				await expect
					.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.expression_enabled)
					.toBe(true);
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});

		/**
		 * Criterion 5 / D-X7. Before X the app's global undo stood down for `INPUT | TEXTAREA | SELECT`,
		 * and the expression field was an `<input>`. It is a contenteditable now, so without
		 * `ui/textEditing.ts` this exact keystroke would have undone the GRAPH — deleting the node the user
		 * was editing — instead of the text. The node surviving is the assertion.
		 */
		test('Ctrl+Z inside the editor undoes text, not the graph', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			try {
				const editor = await fxEditor(page, uid);
				expect(await canUndo(page), 'there IS graph history to destroy').toBe(true);
				await retype(page, editor, '1 + 2');
				await editor.press('Escape'); // close the popup so it does not own the next chord
				await editor.press('Control+z');
				await page.waitForTimeout(300);
				await expect(editor, 'the text changed').not.toHaveText('1 + 2');
				expect(
					await page.evaluate(
						(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
						uid
					),
					'and the node the editor belongs to is still there'
				).toBe(true);
				// The graph stack was not merely spared, it is INTACT: a real undo still reverses the last
				// graph action, which is the fx toggle that opened this editor.
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await undo(page);
				await expect
					.poll(async () => (await nodeParams(page, uid))?.oscillator?.amplitude?.expression_enabled, {
						message: 'the graph history still has the fx toggle to undo'
					})
					.toBe(false);
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});
	});
});

test.describe('the metadata panel', () => {
	/**
	 * THE METADATA TREE OPENS CLOSED, AND ITS HEADER LINE IS A HINT — not the value.
	 *
	 * `MetadataPanel` used to expand every small field on the first frame, so a node with a handful of
	 * meta keys drew a wall of text before the user asked for any of it. Phil's call: collapsed by
	 * default, in BOTH places the component is mounted — the editor's slide-in inspector and the
	 * dockable Metadata panel, which are the same component behind `showHeader`.
	 *
	 * The second test is the one that earns its keep. The collapse choice has to survive the next
	 * frame, and this panel re-renders at the data rate: the `open` state used to be re-asserted from
	 * Svelte state on every frame while the `toggle` event that REPORTS the user's click fires
	 * asynchronously, so a frame landing in that gap silently undid the click. The fix is that the
	 * `<details>` now owns its own open state (collapsed is its own default), which is why this test
	 * provokes a frame it can see rather than waiting on one it hopes for.
	 *
	 * The third pins the header/body split that only the running app can show: both come from the same
	 * meta value, and the two-decimal cap sits on the header alone.
	 *
	 * Runs under the `default` (fine-pointer) project; nothing here is pointer-dependent.
	 */

	/** Add an Oscillator and select it, which slides in the inspector — ParamForm + MetadataPanel. */
	async function selectedOscillator(page: Page): Promise<string> {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, uid);
		await selectNode(page, uid);
		await expect(page.getByTestId('auto-side-panel')).toHaveClass(/open/);
		return uid;
	}

	async function drop(page: Page, uid: string): Promise<void> {
		await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		await waitForNoNode(page, uid).catch(() => {});
	}

	/** The `sfreq` field, addressed by the key it shows rather than by position — the Oscillator grows
	 * a `ufreq` field a second in, which shifts every index after the first. */
	function sfreqField(page: Page, root = page.locator('body')) {
		return root.locator('.meta-field', { has: page.locator('.mk', { hasText: /^sfreq$/ }) });
	}

	test('every metadata field starts collapsed, in the inspector and in the Metadata panel', async ({
		page
	}) => {
		const uid = await selectedOscillator(page);
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		try {
			const inspector = page.getByTestId('auto-side-panel');
			// The panel renders off the live data stream, so the first frame has to land first.
			await expect(
				inspector.locator('.meta-field').first(),
				'a running Oscillator puts at least one meta field on screen'
			).toBeVisible();
			await expect(
				inspector.locator('.meta-field[open]'),
				'the inspector draws no field expanded'
			).toHaveCount(0);

			// The dockable panel is the same component with `showHeader = false`; it takes its node from
			// the panel binding rather than the selection, and must answer the same.
			await page.evaluate(
				([id, u]) => {
					const g = (window as any).goofi.commands;
					g.setPanelType(id, 'metadata');
					g.bindNodeToPanel(id, u);
				},
				[panelId, uid] as const
			);
			await expect(
				page.getByTestId('metadata-slot'),
				'the Metadata panel is showing the bound node'
			).toBeVisible();
			await expect(page.locator('.meta-field').first()).toBeVisible();
			await expect(
				page.locator('.meta-field[open]'),
				'and neither does the Metadata panel'
			).toHaveCount(0);
		} finally {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			await drop(page, uid);
			// AppShell pushes the layout into the RUNNING PATCH on a 400ms debounce, and the patch
			// outlives this page — settle past it so no later spec boots into a metadata-shaped workspace.
			await page.waitForTimeout(700);
		}
	});

	test('a field the user opens stays open when the next frame lands', async ({ page }) => {
		const uid = await selectedOscillator(page);
		try {
			const field = sfreqField(page, page.getByTestId('auto-side-panel'));
			await expect(field, 'the Oscillator tags every frame with its sample rate').toBeVisible();
			await expect(field.locator('.mp')).toHaveText('250');
			await expect(field, 'it starts collapsed, like every field').not.toHaveAttribute('open');

			await field.locator('summary').click();
			await expect(field, 'the click opened it').toHaveAttribute('open', '');

			// Provoke a frame whose meta demonstrably differs, rather than waiting on one that might
			// carry the same text — a re-render the assertion cannot see would make this vacuous.
			await updateParam(page, uid, 'oscillator', 'sfreq', 400);
			await expect(field.locator('.mp'), 'a new frame has rendered').toHaveText('400');
			await expect(field, 'and the user’s choice outlived it').toHaveAttribute('open', '');
		} finally {
			await drop(page, uid);
		}
	});

	test('the header line caps a scalar at two decimals; the expanded body keeps it whole', async ({
		page
	}) => {
		const uid = await selectedOscillator(page);
		try {
			const field = sfreqField(page, page.getByTestId('auto-side-panel'));
			await expect(field).toBeVisible();

			await updateParam(page, uid, 'oscillator', 'sfreq', 333.333333);
			await expect(field.locator('.mp'), 'the header line rounds to two places').toHaveText(
				'333.33'
			);

			await field.locator('summary').click();
			await expect(field.locator('.mv'), 'the body is where the real value lives').toHaveText(
				'333.333333'
			);
		} finally {
			await drop(page, uid);
		}
	});
});
