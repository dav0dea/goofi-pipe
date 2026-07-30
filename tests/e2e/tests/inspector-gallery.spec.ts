import { test, expect } from '@playwright/test';

// The /dev/inspector gallery is a static, backend-free showcase of `<ParamField>` (spec §2, N-Task 2):
// one field per non-expression control kind against synthetic fx-OFF descriptors, each with a
// committed-value read-out. Like the /dev/ui gallery it needs no window.goofi readiness gate — it waits
// on the rendered samples. This spec is the "failing test first": it fails before the route +
// ParamField exist. Runs under the `default` (fine-pointer) project.
test.describe('Inspector field gallery', () => {
	// The D-N2 regression guard (the Critical the spec review caught): vmin/vmax are SOFT bounds — the
	// engine does not clamp on set — so the NumberInput must stay UNBOUNDED. Typing 5 into a [0,1] float
	// commits 5, NOT 1; and the Slider's track auto-extends to span the out-of-range live value.
	test('numeric commits an out-of-[vmin,vmax] value UNCLAMPED and the slider track auto-extends (D-N2)', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-float');
		const number = field.getByTestId('param-number');
		const range = field.getByTestId('param-slider').locator('input[type=range]');
		const readout = page.getByTestId('inspector-float-value');
		await expect(readout, 'seeded 0.3').toHaveText('0.3');
		await expect(range, 'the track is seeded at its soft upper bound').toHaveAttribute('max', '1');
		// Type 5 — well outside the soft [0,1] — and commit it (blur via Enter).
		await number.fill('5');
		await number.press('Enter');
		await expect(readout, 'the out-of-bounds value commits UNCLAMPED (not clipped to vmax)').toHaveText('5');
		await expect(range, 'the slider max auto-extends to span the live value').toHaveAttribute('max', '5');
	});

	// D-N7: the ParamField→NumberInput `scrub` pass-through. Every other numeric case commits via
	// fill()+Enter, leaving the drag-to-scrub forwarding unexercised through the inspector. This drives
	// a real pointer drag on the float sample's NumberInput and asserts the read-out climbs — pinning
	// that ParamField renders the numeric NumberInput with `scrub` enabled. (The scrub arithmetic and
	// pointer lifecycle are already P-tested against the primitive; this only proves the forwarding.)
	test('numeric drag-scrub forwards through ParamField to the NumberInput (D-N7)', async ({ page }) => {
		await page.goto('/dev/inspector');
		const number = page.getByTestId('inspector-float').getByTestId('param-number');
		const readout = page.getByTestId('inspector-float-value');
		await expect(readout, 'seeded 0.3').toHaveText('0.3');
		const box = (await number.boundingBox())!;
		const cx = box.x + box.width / 2;
		const cy = box.y + box.height / 2;
		await page.mouse.move(cx, cy);
		await page.mouse.down();
		await page.mouse.move(cx + 40, cy); // rightward past the 3px threshold → the value climbs
		await page.mouse.up();
		await expect
			.poll(async () => Number(await readout.textContent()), { message: 'the scrub committed a higher value' })
			.toBeGreaterThan(0.3);
	});

	test('numeric (int) commits a typed integer', async ({ page }) => {
		await page.goto('/dev/inspector');
		const number = page.getByTestId('inspector-int').getByTestId('param-number');
		const readout = page.getByTestId('inspector-int-value');
		await expect(readout).toHaveText('5');
		await number.fill('9');
		await number.press('Enter');
		await expect(readout, 'the typed integer commits').toHaveText('9');
	});

	test('toggle flips and commits the boolean', async ({ page }) => {
		await page.goto('/dev/inspector');
		const readout = page.getByTestId('inspector-bool-value');
		await expect(readout).toHaveText('false');
		await page.getByTestId('inspector-bool').getByTestId('param-toggle').click();
		await expect(readout, 'the flipped value commits').toHaveText('true');
	});

	test('trigger commits true on click', async ({ page }) => {
		await page.goto('/dev/inspector');
		const readout = page.getByTestId('inspector-trigger-value');
		await expect(readout, 'no fire yet').toHaveText('0');
		await page.getByTestId('inspector-trigger').getByTestId('param-trigger').click();
		await expect(readout, 'the trigger commits true (the read-out counts a fire)').toHaveText('1');
	});

	test('text commits on Enter (blur), not per keystroke', async ({ page }) => {
		await page.goto('/dev/inspector');
		const input = page.getByTestId('inspector-text').getByTestId('param-text');
		const readout = page.getByTestId('inspector-text-value');
		await expect(readout).toHaveText('hello');
		await input.fill('world');
		// The buffered value has NOT committed while typing — commit-on-blur is inherited from the P control.
		await expect(readout, 'typing buffers; it has not committed yet').toHaveText('hello');
		await input.press('Enter');
		await expect(readout, 'Enter blurs and commits the string').toHaveText('world');
	});

	test('select commits the chosen option through onChange', async ({ page }) => {
		await page.goto('/dev/inspector');
		const readout = page.getByTestId('inspector-options-value');
		await expect(readout, 'seed value shown').toHaveText('sine');
		await page.getByTestId('inspector-options').getByTestId('param-select').locator('select').selectOption('square');
		await expect(readout, 'choosing an option commits it').toHaveText('square');
	});

	// The gate (audit finding #3): a NON-refreshable select renders NO ⟳, even though ParamForm — which
	// this sample mirrors — wires `onRefresh` UNCONDITIONALLY. The ⟳ is gated on `descriptor.refreshable`,
	// not the mere presence of `onRefresh`; before the gate every select showed a ⟳ that the engine would
	// reject by contract.
	test('a non-refreshable select renders NO ⟳ even though onRefresh is wired', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-options');
		// The select itself still renders (and still commits), but there is no refresh affordance.
		await expect(field.getByTestId('param-select').locator('select')).toBeVisible();
		await expect(field.getByTestId('param-refresh'), 'no ⟳ on a non-refreshable param').toHaveCount(0);
		await expect(page.getByTestId('inspector-options-refreshes'), 'onRefresh was never invokable').toHaveText('0');
	});

	// The refreshable select: a stale-but-live current value absent from the options stays selected (P
	// Select prepends it), and the ⟳ button (param-refresh) fires onRefresh, which re-scans the options.
	test('a refreshable select keeps a stale value selected and ⟳ re-scans the options', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-device');
		const select = field.getByTestId('param-select').locator('select');
		const refreshes = page.getByTestId('inspector-device-refreshes');
		// 'mic-1' is not among the seeded options, so its presence as the selection proves the prepend.
		await expect(select, 'the stale-but-live value is still selected').toHaveValue('mic-1');
		await expect(select.locator('option').first(), 'it is prepended as the first option').toHaveText('mic-1');
		await expect(refreshes).toHaveText('0');
		await field.getByTestId('param-refresh').click();
		await expect(refreshes, 'the ⟳ button fired onRefresh').toHaveText('1');
		// The re-scan lands a fresh option the seed list did not have.
		await expect(select.locator('option'), 'the option list was refreshed').toContainText(['scanned-1']);
	});

	test('unknown renders a read-only JSON display of the value', async ({ page }) => {
		await page.goto('/dev/inspector');
		const display = page.getByTestId('inspector-unknown').getByTestId('param-unknown');
		// The raw JSON of the synthetic descriptor value — a read-only fallback, no editable control.
		await expect(display).toContainText('"channels"');
		await expect(display).toContainText('Fz');
	});
});

// The fx (expression) binding (spec §3, D-N3/D-N4, N-Task 3): the always-present fx Chip adornment, the
// trig Chip (only when active), the buffered single-line expr input, the IN-PANEL multi-line editor (no
// modal), and the three enabled-semantics. The `inspector-fx` sample starts fx-OFF with all expression
// state exposed as read-outs (expr / enabled / triggers) so every toggle is observable; `inspector-fx-error`
// is a fixed errored expression. Runs under the `default` (fine-pointer) project.
test.describe('Inspector fx (expression) binding', () => {
	// Enabled-semantic #1 (OFF→ON seeds a literal) + the fx-off stash: toggling on seeds the current value
	// as a Python literal and enables; toggling back off KEEPS the source stashed (enabled:false).
	test('fx toggles on (seeding a literal) and off (stashing the source)', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		const fx = field.getByTestId('param-fx-toggle');
		const expr = page.getByTestId('inspector-fx-expr');
		const enabled = page.getByTestId('inspector-fx-enabled');
		await expect(enabled, 'starts fx-off').toHaveText('false');
		await expect(expr, 'no source yet').toHaveText('null');
		await fx.click();
		await expect(enabled, 'fx-on enables the expression').toHaveText('true');
		await expect(expr, 'OFF→ON seeds the current value 0.5 as a Python literal').toHaveText('0.5');
		await expect(field.getByTestId('param-expr-input'), 'the expr input takes over the control region').toBeVisible();
		await fx.click();
		await expect(enabled, 'fx-off disables the engine').toHaveText('false');
		await expect(expr, 'ON→OFF stashes the source so a later flip-on does not retype').toHaveText('0.5');
	});

	// The trig chip: hidden while fx is off, appears with fx active, and toggles process-on-change.
	test('the trig chip is fx-gated and toggles process-on-change', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		await expect(field.getByTestId('param-expr-triggers-process'), 'no trig chip while fx is off').toHaveCount(0);
		await field.getByTestId('param-fx-toggle').click();
		const trig = field.getByTestId('param-expr-triggers-process');
		await expect(trig, 'trig appears once fx is active').toBeVisible();
		await expect(page.getByTestId('inspector-fx-triggers')).toHaveText('false');
		await trig.click();
		await expect(page.getByTestId('inspector-fx-triggers'), 'trig toggles process-on-change').toHaveText('true');
	});

	/* Both adornment Chips are two-state toggles, so both must SAY so. The pre-overhaul inspector
	   declared `aria-pressed` on each; the rewrite passed tone/onclick/title/testid and dropped it,
	   even though `Chip` spreads `...rest` and would have forwarded it. `trig` is the clean loss —
	   its title is static, so its state lived in the tone alone, which is to say in colour alone.
	   The convention is intact everywhere else in the app (ConsolePanel's out/err use the same Chip
	   with the same shape), which is what makes this a regression rather than a decision. */
	test('the fx and trig chips announce their pressed state', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		const fx = field.getByTestId('param-fx-toggle');
		await expect(fx, 'fx rests unpressed').toHaveAttribute('aria-pressed', 'false');
		await fx.click();
		await expect(fx, 'fx reports itself pressed once active').toHaveAttribute('aria-pressed', 'true');

		const trig = field.getByTestId('param-expr-triggers-process');
		await expect(trig, 'trig rests unpressed').toHaveAttribute('aria-pressed', 'false');
		await trig.click();
		await expect(trig, 'trig reports itself pressed').toHaveAttribute('aria-pressed', 'true');
	});

	// Enabled-semantic #2 (inline commit FORCES enabled): editing the single-line buffer and committing
	// (Enter blurs) writes the new source and keeps the expression enabled.
	test('committing the inline buffer writes the source and keeps it enabled', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		await field.getByTestId('param-fx-toggle').click();
		const input = field.getByTestId('param-expr-input');
		await input.fill("nd('a').out.data.mean()");
		await input.press('Enter');
		await expect(page.getByTestId('inspector-fx-expr'), 'the committed buffer becomes the source').toHaveText("nd('a').out.data.mean()");
		await expect(page.getByTestId('inspector-fx-enabled'), 'commit forces enabled:true').toHaveText('true');
	});

	// D-N4 (in-panel, no modal) + enabled-semantic #3 (apply PRESERVES flags): expand grows the input into
	// an in-panel textarea; ⌘/Ctrl+Enter applies the source and collapses back to the single line.
	test('expand opens an in-panel multi-line editor; Ctrl+Enter applies and collapses', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		await field.getByTestId('param-fx-toggle').click();
		await field.getByTestId('param-expr-expand').click();
		const ta = field.getByTestId('param-expr-multiline');
		await expect(ta, 'the editor grows in place — no modal').toBeVisible();
		await expect(field.getByTestId('param-expr-input'), 'the single-line input is replaced while expanded').toHaveCount(0);
		await ta.fill("nd('b').out.data.std()");
		await ta.press('Control+Enter');
		await expect(ta, 'apply collapses the multi-line editor').toHaveCount(0);
		await expect(page.getByTestId('inspector-fx-expr'), 'apply commits the source (flags preserved)').toHaveText("nd('b').out.data.std()");
		await expect(field.getByTestId('param-expr-input'), 'the single-line input returns').toBeVisible();
	});

	/* The `apply` Chip is the door touch has to the ⌃⏎ chord, and since X it takes a different route:
	   the editor OWNS its document (a CodeMirror document is not a bindable string), so the Chip asks it
	   to commit through `ExprEditor`'s `bindCommit` seam instead of reading a mirrored buffer. Nothing
	   covered this control before, and the seam is new — so it is covered now. */
	test('the apply Chip commits through the editor and collapses', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		await field.getByTestId('param-fx-toggle').click();
		await field.getByTestId('param-expr-expand').click();
		const ta = field.getByTestId('param-expr-multiline');
		await ta.fill("nd('c').out.data.max()");
		await field.getByTestId('param-expr-apply').click();
		await expect(ta, 'apply collapses the editor').toHaveCount(0);
		await expect(page.getByTestId('inspector-fx-expr'), 'the Chip committed the editor’s document').toHaveText("nd('c').out.data.max()");
		await expect(page.getByTestId('inspector-fx-enabled'), 'apply PRESERVES the flags').toHaveText('true');
	});

	/* Escape reverts + collapses the multi-line editor without committing — and since X it is LAYERED
	   (D-X7): a completion popup owns the first Escape, the editor the next. Typing `ME` leaves a popup
	   open, which is worth asserting for its own sake: the only thing matching it is Python's own
	   `MemoryError` builtin, so this is the stock language sources answering inside our editor. */
	test('Escape closes the completion first, then reverts and collapses the editor', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		await field.getByTestId('param-fx-toggle').click();
		await field.getByTestId('param-expr-expand').click();
		const ta = field.getByTestId('param-expr-multiline');
		await ta.fill('DISCARD ME');
		const popup = page.locator('.cm-tooltip-autocomplete');
		await expect(popup, 'Python’s own builtins answer for `ME`').toBeVisible();
		await ta.press('Escape');
		await expect(popup, 'the first Escape belongs to the popup').toHaveCount(0);
		await expect(ta, 'and the editor is still open').toHaveCount(1);
		await ta.press('Escape');
		await expect(ta, 'the next Escape collapses the editor').toHaveCount(0);
		await expect(page.getByTestId('inspector-fx-expr'), 'Escape reverts — the source is unchanged (no commit)').toHaveText('0.5');
	});

	// An errored expression: the fx Chip carries the danger tone (a pointer) and the full message renders in
	// the param-expr-error row.
	test('an expression error reddens the fx chip and renders the error row', async ({ page }) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx-error');
		await expect(field.getByTestId('param-fx-toggle'), 'the fx chip carries the danger tone').toHaveClass(/t-danger/);
		const err = field.getByTestId('param-expr-error');
		await expect(err, 'the error row renders below the input').toBeVisible();
		await expect(err).toContainText('NameError');
	});
});

// ParamForm (spec §4, N-Task 4): the node-driven assembler — identity header + inline rename + docs
// Disclosure, the connected P Tabs bar it now OWNS, and show_when field/tab filtering — mounted in the
// gallery against a synthetic multi-group node (`inspector-form`) with a showWhen prop that hides
// `depth`/`cutoff` unless mode==='filter'. ParamForm binds the real store; in this backend-free gallery
// the in-form commits are inert, so the controlling `mode` value is driven through a gallery-owned
// toggle (`form-toggle-mode`) — proving the show_when reactive filter + the all-hidden-group tab rule.
// Runs under the `default` (fine-pointer) project.
test.describe('Inspector ParamForm', () => {
	const bg = (loc: import('@playwright/test').Locator) =>
		loc.evaluate((el) => getComputedStyle(el).backgroundColor);

	test('owns the connected group tabs — the active tab drops to the body surface, and tabs switch groups', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		const form = page.getByTestId('inspector-form');
		const tabs = form.getByTestId('param-tabs');
		// `common` sorts LAST; `filter` is all-hidden while mode=pass, so the first visible tab is `signal`.
		const active = tabs.getByRole('tab', { selected: true });
		await expect(active, 'the first visible group is active').toHaveText('signal');
		// Connected: the active tab paints the SAME surface as the rows body flush beneath it (merged);
		// an inactive tab sits at the header surface, a different colour (the drop actually happened).
		const rowsBg = await bg(form.getByTestId('param-rows'));
		const activeBg = await bg(active);
		const inactiveBg = await bg(tabs.getByRole('tab', { selected: false }).first());
		expect(activeBg, 'active tab background equals the body surface (merged)').toBe(rowsBg);
		expect(inactiveBg, 'an inactive tab sits at the header surface, not the body').not.toBe(activeBg);
		// Switching tabs swaps the rendered fields: `signal` shows `gain`; `common` shows `label`.
		await expect(form.getByTestId('param-field-gain')).toBeVisible();
		await expect(form.getByTestId('param-field-label')).toHaveCount(0);
		await tabs.getByRole('tab', { name: 'common' }).click();
		await expect(form.getByTestId('param-field-label'), 'the common tab renders its field').toBeVisible();
		await expect(form.getByTestId('param-field-gain'), 'the signal field is gone').toHaveCount(0);
	});

	test('show_when hides a dependent field and its all-hidden group tab, then reveals both live', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		const form = page.getByTestId('inspector-form');
		const tabs = form.getByTestId('param-tabs');
		// mode=pass: `depth` (a same-group signal dependent) is hidden, and the `filter` group — whose
		// only field `cutoff` is hidden — renders NO tab (never a live tab over an all-hidden group).
		await expect(form.getByTestId('param-field-depth'), 'the dependent field is hidden').toHaveCount(0);
		await expect(tabs.getByRole('tab', { name: 'filter' }), 'the all-hidden group shows no tab').toHaveCount(0);
		// Flip the controlling `mode` to `filter`; show_when re-evaluates live off the node's live values.
		await page.getByTestId('form-toggle-mode').click();
		await expect(form.getByTestId('param-field-depth'), 'the same-group dependent appears live, no tab switch').toBeVisible();
		await expect(tabs.getByRole('tab', { name: 'filter' }), 'the refilled group tab appears').toBeVisible();
		// The revealed cross-group dependent renders once its tab is selected (liveValues cross groups).
		await tabs.getByRole('tab', { name: 'filter' }).click();
		await expect(form.getByTestId('param-field-cutoff'), 'the cross-group dependent renders under its tab').toBeVisible();
		// Flip back to `pass`: the filter group empties → its tab disappears AND the active group falls
		// back to a still-visible tab (activeGroup revalidation), never leaving `filter` active.
		await page.getByTestId('form-toggle-mode').click();
		await expect(tabs.getByRole('tab', { name: 'filter' }), 'the emptied group tab disappears').toHaveCount(0);
		await expect(
			tabs.getByRole('tab', { selected: true }),
			'the active group falls back off the vanished filter tab'
		).not.toHaveText('filter');
		await expect(form.getByTestId('param-field-depth'), 'the same-group dependent hides again live').toHaveCount(0);
	});

	test('the docstring hides behind a Disclosure that toggles', async ({ page }) => {
		await page.goto('/dev/inspector');
		const form = page.getByTestId('inspector-form');
		await expect(form.getByTestId('docstring'), 'the docstring starts collapsed').toHaveCount(0);
		await form.getByTestId('docs-toggle').click();
		await expect(form.getByTestId('docstring'), 'the disclosure reveals the docstring').toBeVisible();
		await form.getByTestId('docs-toggle').click();
		await expect(form.getByTestId('docstring'), 'toggling again collapses it').toHaveCount(0);
	});

	// The inline rename null-first commit/cancel dance: Escape reverts (no commit); Enter runs the commit
	// path then closes the editor (the unmount-blur it triggers is a null-guarded no-op — no double-commit).
	test('inline rename opens on click; Escape reverts and Enter commits then closes the editor', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		const form = page.getByTestId('inspector-form');
		await expect(form.getByTestId('node-name'), 'the display name shows').toHaveText('oscillator0');
		// Open the editor; the draft seeds from the current name.
		await form.getByTestId('node-name').click();
		await expect(form.getByTestId('node-name-input')).toHaveValue('oscillator0');
		// Escape cancels — the editor closes and the name is unchanged (no commit).
		await form.getByTestId('node-name-input').fill('discard-me');
		await form.getByTestId('node-name-input').press('Escape');
		await expect(form.getByTestId('node-name-input'), 'Escape closes the editor').toHaveCount(0);
		await expect(form.getByTestId('node-name'), 'the name is unchanged on cancel').toHaveText('oscillator0');
		// Enter runs the commit path (null-first) and closes the editor — the read-only name returns.
		await form.getByTestId('node-name').click();
		await form.getByTestId('node-name-input').fill('renamed');
		await form.getByTestId('node-name-input').press('Enter');
		await expect(form.getByTestId('node-name-input'), 'Enter closes the editor after committing').toHaveCount(0);
		await expect(form.getByTestId('node-name'), 'the read-only name button returns').toBeVisible();
	});
});
