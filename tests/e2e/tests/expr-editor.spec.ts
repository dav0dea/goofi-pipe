import { test, expect, type Locator, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import {
	addNode,
	addGlobal,
	waitForNode,
	waitForNoNode,
	nodeParams,
	canUndo,
	undo
} from '../lib/goofi';

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
	await page.evaluate((u) => (window as any).goofi.commands.select([u]), uid);
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
