// The app frame around the workspace: boot, the top bar, and the shared visual primitives
// every page inherits.

import { test, expect, type Page, type Locator } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { waitForApp, resetPatch } from '../lib/app';
import { addErroringNode, addNode, selectNode, waitForNoNode, waitForNode } from '../lib/goofi';
import { AS_ROWS, PRIORITY, inBar, menuRow, openOverflow, settledBar } from '../lib/topbar';
import { kbInset, setKeyboardInset } from '../lib/touch';

test.describe('the app boots', () => {
	test('boots the backend and the app connects + loads the catalog', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// The catalog arrived over the control WS: Oscillator + Buffer are always present.
		const types = await page.evaluate(() => (window as any).goofi.query.nodeTypes());
		expect(Array.isArray(types)).toBe(true);
		expect(types.map((t: any) => t.type)).toContain('Oscillator');
	});

	test('a tab reached from somewhere else still reloads', async ({ page, baseURL }) => {
		// The drive-by guard refused every cross-site request, and a top-level NAVIGATION is one.
		// The browser then replays that classification on every later reload of the tab, so a
		// window opened from a link — or from another port — answered 403 for ever, and the only
		// way back in was to retype the address. Restarting goofi and hitting reload is exactly
		// when a user meets it, which is why this drives a real browser rather than a header.
		//
		// `localhost` and `127.0.0.1` are the same machine and DIFFERENT sites to Chromium, so the
		// same backend serves both and the hop between them is genuinely cross-site.
		const url = new URL(baseURL!);
		await page.goto(`http://localhost:${url.port}/`);
		await waitForApp(page);

		await page.evaluate((to) => window.location.assign(to), baseURL!);
		await waitForApp(page);
		expect(new URL(page.url()).hostname, 'the hop landed on the other site').toBe('127.0.0.1');

		// …and again, because the replay is what made it stick.
		await page.reload();
		await waitForApp(page);
		await page.reload();
		await waitForApp(page);
	});
});

test.describe('the top bar', () => {
	/**
	 * What the app header is allowed to carry.
	 *
	 * The header is the only constant chrome, so it holds app-global actions: undo/redo of the whole
	 * session, and the patch's save/load. It used to also hold ＋ Add node and Fit, which are neither —
	 * both act on ONE node-editor panel (they resolve "the active editor" behind the user's back, and
	 * silently pick an arbitrary one when several are open), and every editor panel already offers
	 * both locally: the flow controls' own fit button, and four add-node doors of its own.
	 *
	 * Pinned as an exact ordered list rather than two absence assertions, so the header's contents and
	 * their reading order are both fixed. This is the DOM order, which is NOT the priority order
	 * D-R6's progressive overflow spills from — that is Undo · Redo · Save · Load… · Save▾ (the caret
	 * leaves first, and the split degrades into a plain Save button), and `topbar-overflow` — the
	 * resident menu the spilled actions land in — deliberately lives outside `.actions`.
	 * `topbar-overflow.spec.ts` owns the spill order.
	 */
	const HEADER_ACTIONS = [
		'topbar-undo',
		'topbar-redo',
		'topbar-save',
		'topbar-save-caret',
		'topbar-load'
	];

	test('the app header follows the fine-pointer control height', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const bar = (await page.locator('.topbar').boundingBox())!;
		const control = (await page.getByTestId('topbar-overflow').boundingBox())!;
		expect(bar.height, 'the bar adds no fixed vertical slack around its tallest control').toBe(
			control.height
		);
		expect(bar.height, 'desktop chrome does not reserve the touch-sized height').toBeLessThan(44);
	});

	test('the app header carries exactly the app-global actions', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const ids = await page
			.locator('.topbar .actions button')
			.evaluateAll((els) => els.map((el) => el.getAttribute('data-testid')));
		expect(ids, 'panel-local behaviour does not belong in the app header').toEqual(HEADER_ACTIONS);
	});

	test('the header is one icon family on one chrome text size (Phil: de-clutter)', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		// Every bar action is a GLYPH with an accessible name — Save/Load joined undo/redo, so the
		// action cluster reads as one family instead of two idioms side by side. The words live on
		// in the tooltips and the overflow-menu rows.
		for (const id of ['topbar-undo', 'topbar-redo', 'topbar-save', 'topbar-load']) {
			const btn = page.getByTestId(id);
			expect(
				(await btn.textContent())?.trim(),
				`${id} carries a glyph, not a word`
			).toBe('');
			await expect(btn.locator('svg'), `${id} renders its icon`).toHaveCount(1);
			expect(await btn.getAttribute('aria-label'), `${id} keeps its name for AT`).toBeTruthy();
		}

		// The bar's remaining TEXT (the tab labels, the patch name, the fps) shares ONE chrome
		// size — integer-snapped, so every baseline in the bar lands on the same device row.
		// The fluid clamp's fractional sizes snapped per element and scattered the ink ±0.75px.
		const sizeOf = (sel: string): Promise<string> =>
			page.locator(sel).first().evaluate((el) => getComputedStyle(el).fontSize);
		const tab = await sizeOf('[data-testid="workspace-tabs"] .ui-tab-label');
		expect(parseFloat(tab) % 1, `the chrome size (${tab}) snaps to whole pixels`).toBe(0);

		// The name chip and the fps readout render once the patch is live — adding a node dirties
		// it (● untitled) and starts frames flowing, which is both preconditions at once.
		const uid = await page.evaluate(() =>
			(window as any).goofi.commands.addNode('Oscillator', 'inputs', [40, 40])
		);
		try {
			const path = page.getByTestId('topbar-path');
			await path.waitFor({ state: 'attached' });
			expect(
				await path.evaluate((el) => getComputedStyle(el).fontSize),
				'the patch name shares the chrome size'
			).toBe(tab);

			// The fps readout is quiet text like the name beside it — the boxed pill is gone.
			const hud = page.getByTestId('perf-hud');
			await hud.waitFor({ state: 'attached' });
			const skin = await hud.evaluate((el) => {
				const cs = getComputedStyle(el);
				return { bg: cs.backgroundColor, fs: cs.fontSize };
			});
			expect(
				/rgba\(0, 0, 0, 0\)|transparent/.test(skin.bg),
				`the fps readout (${skin.bg}) is unboxed`
			).toBe(true);
			expect(skin.fs, 'the fps readout shares the chrome size').toBe(tab);
			// The HUD is the paint rate ALONE. The drop counter that used to sit here summed coalesced
			// frames across every stream, which put a total beside a number that is not one; a drop
			// belongs to the stream whose frame was overwritten, so it moved to the Metadata panel
			// beside that node's update rate.
			await expect(hud).toHaveText(/^\s*\d+ fps\s*$/);
			await expect(hud.getByTestId('perf-drops')).toHaveCount(0);

			// Identity and action items share one overflow group and one visual rhythm. Measure the INK
			// rather than the outer boxes: the text items deliberately borrow the clear space that each
			// IconButton's square puts around its glyph.
			const rhythm = await page.locator('.topbar .actions').evaluate((group) => {
				const box = (selector: string): DOMRect =>
					group.querySelector(selector)!.getBoundingClientRect();
				const fps = box('[data-testid="perf-hud"]');
				const name = box('.path-value');
				const undo = box('[data-testid="topbar-undo"] svg');
				const redo = box('[data-testid="topbar-redo"] svg');
				return {
					members: [
						'topbar-hud',
						'topbar-path',
						'topbar-undo',
						'topbar-redo',
						'topbar-save',
						'topbar-save-caret',
						'topbar-load'
					].every((id) => group.querySelector(`[data-testid="${id}"]`)),
					gaps: [name.left - fps.right, undo.left - name.right, redo.left - undo.right]
				};
			});
			expect(rhythm.members, 'fps, name and icon actions live in one spillable group').toBe(true);
			expect(Math.min(...rhythm.gaps), 'no identity ink sits flush against its neighbour').toBeGreaterThan(
				12
			);
			expect(
				Math.max(...rhythm.gaps) - Math.min(...rhythm.gaps),
				`text/text, text/icon and icon/icon gaps share one rhythm (${rhythm.gaps.join(', ')})`
			).toBeLessThanOrEqual(2);

			// The HUD is not merely in the same flex box; it participates in the same relocation plan.
			await page.setViewportSize({ width: 320, height: 720 });
			await expect(page.getByTestId('topbar-hud'), 'the FPS item spills as the group narrows').toBeHidden();
			await page.getByTestId('topbar-overflow').click();
			const fpsRow = page.getByRole('menuitem', { name: /^\d+ fps$/ });
			await expect(fpsRow, 'the same FPS information moves into the overflow menu').toHaveCount(1);
			await expect(fpsRow, 'information relocates without becoming an action').toBeDisabled();
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
		}
	});

	test('the save menu offers Save As and nothing else', async ({ page }) => {
		// "Save in browser" is gone (user decision, 2026-08-08): a save writes a backend file, full
		// stop. The caret stays a menu so the split control keeps its shape and its spill behaviour.
		await page.goto('/');
		await waitForApp(page);
		await page.getByTestId('topbar-save-caret').click();
		const rows = page.locator('.context-menu [role="menuitem"]');
		await expect(rows, 'one row: Save As…').toHaveCount(1);
		await expect(rows.first()).toHaveText(/Save As/);
		await expect(
			page.locator('.context-menu').getByText(/browser/i),
			'no browser-save row anywhere in it'
		).toHaveCount(0);
	});

	/**
	 * …and what it is not allowed to carry: a brand.
	 *
	 * Phil's call, and both halves of it are the same point. The ⟁ is not goofi's logo — it never was —
	 * and the wordmark spends ~95px of a 412px bar restating what the browser tab already says, in the
	 * one strip of chrome that is on screen at every width. The bar is for what the user can DO here.
	 *
	 * Read off `textContent`, not `innerText`: the wordmark was hidden below 520px by a container
	 * query, so a visible-text assertion would already have been green on a phone while the brand was
	 * still in the DOM taking part in the layout.
	 */
	test('the app header carries no brand', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const text = (await page.locator('.topbar').textContent()) ?? '';
		expect(text, 'the wordmark is the browser tab’s job').not.toContain('goofi-pipe');
		expect(text, 'and the ⟁ was never the logo').not.toContain('⟁');
	});
});

test.describe('a top bar that does not fit', () => {
	/**
	 * The app header's progressive overflow (D-R6) and the canvas commands it carries (D-R4).
	 *
	 * This is a DESKTOP spec on purpose. The collapse keys on available WIDTH, not on device class —
	 * a 900px desktop window has exactly the phone's problem — and the menu is resident chrome at
	 * every width, because the commands inside it (delete, group, select-all, copy/paste/duplicate,
	 * multi-select) have no bar slot to lose and are otherwise reachable only by a keyboard chord.
	 *
	 * The three traps D-R6 names are pinned as arithmetic in `editor/overflowFit.test.ts`; what this
	 * file proves is that the real bar is wired to that arithmetic — that items really leave in the
	 * declared order, that the result is stable when the width crosses a boundary and comes back, and
	 * that nothing the header used to do changed.
	 */

	/** The bar gives its actions up in reverse priority order. (`PRIORITY`, `AS_ROWS` and the
	 * handles for reading the bar live in `lib/topbar.ts` — `touch-reflow.spec.ts` asks the same
	 * question of the same list at three real device geometries.) */
	const SPILL_ORDER = [...PRIORITY].reverse();

	/**
	 * How many actions spill depends on the status cluster's width, and that depends on whether
	 * the patch has a NAME — which this file never established. It reached three only because
	 * `dirty-taxonomy.spec.ts` and `fs-browser.spec.ts` sort earlier in the single-worker `default`
	 * project and left the backend in a state nothing here asked for; standalone it measured two. So
	 * every test that needs a spill to EXIST brings its own precondition and hands the backend back
	 * unnamed (`resetPatch` — a manager-side `new` — is what resets `save_path` to null).
	 *
	 * That is four tests now: two of them joined the list when the header stopped carrying a connection
	 * chip. "Connected" was up at every moment of the app's life, and its 72px was quietly what pushed
	 * an untitled patch's bar over the edge at 320px. Nothing about the overflow changed — the bar
	 * simply has 72px more to work with, so width alone no longer guarantees the spill those two are
	 * written to observe.
	 *
	 * The rest of the file is order-independent on purpose: declared order and bar/menu parity are
	 * statements about a bar at whatever width, not about a particular one.
	 */
	const CROWDING_NAME = 'a-patch-with-a-deliberately-long-name-that-crowds-the-header';
	let scratch = '';

	test.beforeAll(() => {
		scratch = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'goofi-e2e-overflow-')));
	});
	test.afterAll(() => fs.rmSync(scratch, { recursive: true, force: true }));

	async function withNamedPatch(page: Page, body: () => Promise<void>): Promise<void> {
		const file = path.join(scratch, `${CROWDING_NAME}.gfi`);
		await page.evaluate((p) => (window as any).goofi.commands.save(p), file);
		await expect(page.locator('.topbar .path'), 'the header is showing the patch name').toHaveText(
			`${CROWDING_NAME}.gfi`
		);
		try {
			await body();
		} finally {
			await resetPatch(page);
		}
	}

	/** Resize and let the ResizeObserver settle (it runs after layout, before the next paint). */
	async function widthTo(page: Page, width: number): Promise<void> {
		await page.setViewportSize({ width, height: 720 });
		await page.evaluate(
			() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
		);
	}

	test('the overflow trigger is resident chrome, outside the pinned action list', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const trigger = page.getByTestId('topbar-overflow');
		await expect(trigger).toBeVisible();
		// `topbar.spec.ts` pins `.topbar .actions button` as an exact ordered list of the five
		// app-global actions. The trigger is not one of them, so it must not live in that box.
		await expect(page.locator('.topbar .actions [data-testid="topbar-overflow"]')).toHaveCount(0);
	});

	test('a wide window keeps every action in the bar', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 1400);
		expect((await inBar(page)).sort()).toEqual([...PRIORITY].sort());
	});

	test('actions leave one at a time, lowest priority first — and the identity leaves before them', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		await withNamedPatch(page, async () => {
			const left: string[] = [];
			let prev = new Set<string>(PRIORITY);
			let pathLeftAt = 0;
			let firstActionLeftAt = 0;
			// Down to 200 (was 280 when Save/Load carried words): every action still fits at 280 now
			// that all five are icons, and the walk must reach a width where the bar genuinely gives
			// actions up — probed: the caret leaves below 280, Load below 240, Save below 220.
			for (let w = 1400; w >= 200; w -= 20) {
				await widthTo(page, w);
				const now = new Set(await inBar(page));
				for (const id of now) {
					expect(prev.has(id), `${id} came BACK into the bar at ${w}px`).toBe(true);
				}
				for (const id of prev) if (!now.has(id)) left.push(id);
				if (!firstActionLeftAt && left.length) firstActionLeftAt = w;
				if (!pathLeftAt && (await page.getByTestId('topbar-path').isHidden())) pathLeftAt = w;
				prev = now;
			}
			expect(left.length, 'the bar does give actions up as it narrows').toBeGreaterThanOrEqual(1);
			expect(left, 'and it gives them up in the declared priority order').toEqual(
				SPILL_ORDER.slice(0, left.length)
			);
			expect(pathLeftAt, 'the patch identity yielded on the way').toBeGreaterThan(0);
			expect(
				pathLeftAt,
				'…and it yielded BEFORE the first action did (informational outranks nothing)'
			).toBeGreaterThan(firstActionLeftAt);
		});
	});

	test('every spilled action is reachable in the overflow menu — and only those', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await withNamedPatch(page, async () => {
			// 240 (was 280 when Save/Load carried words): the icon-only actions are ~130px narrower,
			// so the mixed state this test wants (some actions in the bar, some in the menu, identity
			// already gone) starts a step lower — probed: 240 keeps undo/redo/save and spills the
			// caret and Load.
			await widthTo(page, 240);
			// SETTLED, not a single read: at a named patch the plan converges over several observer
			// rounds (the cluster and the strip give way together), and a read taken mid-flight reports
			// an action as kept one frame before it is `display: none` — which then fails against the
			// menu, where it has already arrived. Caught at --repeat-each=20, ~10% of runs.
			const kept = await settledBar(page);
			expect(kept.length, 'something spilled at 280px').toBeLessThan(PRIORITY.length);
			await expect(
				page.getByTestId('topbar-path'),
				'the identity spilled long before this width'
			).toBeHidden();

			await openOverflow(page);
			for (const id of PRIORITY) {
				const spilledHere = !kept.includes(id);
				for (const label of AS_ROWS[id]) {
					// Present exactly when the bar gave it up: a row that duplicates a visible button is
					// two doors onto one action, which is how the two representations drift apart.
					await expect(menuRow(page, label), `${label} (${id} spilled: ${spilledHere})`).toHaveCount(
						spilledHere ? 1 : 0
					);
				}
			}
		});
	});

	/* Trap 1, at the real bar rather than at the arithmetic: moving an item out changes the bar's
	   content, which re-fires the observer that moved it. A plan that read the bar's own width would
	   flip-flop forever at exactly the boundary. */
	test('the bar is stable across a boundary width, in both directions', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await withNamedPatch(page, async () => {
			// Find the widest width at which an ACTION spills, then straddle it. Down to 260: the
			// identity yields first, so the action boundary sits lower than it used to.
			//
			// BISECTED on the same 4px grid, rather than walked down it: 285 resizes at ~52ms each was
			// 4.9% of the whole suite in this one test, and ~10 probes answer the same question. What
			// makes it sound is the invariant the linear walk already assumed — spill is monotone in
			// width, which is the only reason stopping at the first one and calling it THE boundary was
			// ever legitimate — plus the last test in this file, which is the one that proves a width has
			// one answer however it is approached, so a probe that jumps reads the same bar a step would.
			// `settledBar`, not `inBar`: a jump re-plans harder than a 4px step, and a single read taken
			// mid-flight is how this file has been flaky before. The loop exits with `lo` spilling and
			// `hi === lo + 4` not, so non-spill just above the boundary is proven on the way in.
			const spills = async (w: number): Promise<boolean> => {
				await widthTo(page, w);
				return (await settledBar(page)).length < PRIORITY.length;
			};
			expect(await spills(260), 'the bar does overflow somewhere between 260 and 1400').toBe(true);
			expect(await spills(1400), '…and 1400px still holds every action').toBe(false);
			let lo = 260;
			let hi = 1400;
			while (hi - lo > 4) {
				const mid = lo + 4 * Math.floor((hi - lo) / 8); // on the grid, strictly between the two
				if (await spills(mid)) lo = mid;
				else hi = mid;
			}
			const boundary = lo;

			await widthTo(page, boundary);
			const at = (await inBar(page)).join();
			await widthTo(page, boundary + 24);
			const above = (await inBar(page)).join();
			for (let i = 0; i < 4; i++) {
				await widthTo(page, boundary);
				expect(await inBar(page), 'crossing back lands on the same answer').toEqual(at.split(','));
				await widthTo(page, boundary + 24);
				expect(await inBar(page)).toEqual(above.split(','));
			}

			// …and having settled, it stays settled with nobody touching it.
			await widthTo(page, boundary);
			await page.waitForTimeout(500);
			expect(await inBar(page)).toEqual(at.split(','));
		});
	});

	test('the layout tab strip keeps a floor to spill against', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await widthTo(page, 412);
		// o1: the actions used to take every pixel and squeeze `.tabslot` to zero, so every layout tab
		// AND the ＋ that makes one were unreachable. Giving the actions a floor to spill below is
		// what fixes it.
		const slot = await page.locator('.topbar .tabslot').boundingBox();
		expect(slot!.width, 'the tab strip is not squeezed to nothing').toBeGreaterThan(40);
		await expect(page.getByRole('button', { name: 'New tab' })).toBeVisible();
	});

	/**
	 * R's §5.2 for the chord half: no interaction may exist solely behind a keyboard chord. This list
	 * is the editor's chord inventory (`NodeEditorPanel.onKeydown`) minus the three that already have
	 * a pointer door of their own — Tab (long-press the canvas), F (the panel's own Fit control) and
	 * Escape (a tap on the canvas, or the inspector's ✕). Everything else is here, at every width,
	 * because the menu is resident chrome rather than a collapse target.
	 */
	const CHORD_ROWS = [
		'Select all', // ⌘A
		'Delete selection', // Delete
		'Group into sub-patch', // ⌘G
		'Copy', // ⌘C
		'Paste', // ⌘V
		'Duplicate', // ⌘D
		'Multi-select mode' // no chord at all — the touch door for shift-click
	];

	test('the canvas commands live in the menu at every width', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		for (const w of [1400, 412]) {
			await widthTo(page, w);
			await openOverflow(page);
			for (const name of CHORD_ROWS) {
				await expect(menuRow(page, name), `${name} at ${w}px`).toBeVisible();
			}
			await page.keyboard.press('Escape');
		}
	});

	test('Select all from the menu selects the editor’s nodes', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await page.evaluate(() =>
			(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
		);
		await page.waitForFunction(
			(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
			uid
		);
		try {
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await openOverflow(page);
			await menuRow(page, 'Select all').click();
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
				.toBeGreaterThan(0);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
		}
	});

	test('multi-select mode is a mode: it stays on, and the header says so', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const trigger = page.getByTestId('topbar-overflow');
		// The trigger says the ONE thing it owns and changes on activation: whether the menu is open.
		// The mode is not its state — it neither owns it nor toggles it — so it carries the mode only
		// as the visible tell (the accent + the title), and the row that toggles it says `aria-checked`.
		await expect(trigger).toHaveAttribute('aria-expanded', 'false');

		await openOverflow(page);
		await expect(trigger).toHaveAttribute('aria-expanded', 'true');
		const off = menuRow(page, 'Multi-select mode');
		await expect(off, 'a checkable row is a checkbox, not a plain item').toHaveAttribute(
			'aria-checked',
			'false'
		);
		// The check mark and the per-row icons are decoration: the row's NAME is its label alone.
		await expect(off).toHaveAccessibleName('Multi-select mode');
		await off.click();
		await expect(trigger, 'the always-visible chrome carries the mode').toHaveClass(/multi-on/);
		await expect(trigger).toHaveAttribute('title', /multi-select mode is on/);

		// …and the row itself reads back as checked next time the menu is opened.
		await openOverflow(page);
		const row = menuRow(page, 'Multi-select mode');
		await expect(row.locator('.check svg')).toHaveAttribute('data-icon', 'check');
		await expect(row).toHaveAttribute('aria-checked', 'true');
		await expect(row).toHaveAccessibleName('Multi-select mode');
		await row.click();
		await expect(trigger).not.toHaveClass(/multi-on/);
	});

	/* One label column per menu. `.check` used to render on every row while `.ic` rendered only where
	   there was an icon, so the one menu that mixes them — this one — laid its labels out in two
	   columns 18px apart, on the phone and on the desktop alike. */
	test('every row in a menu starts its label in the same column', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await openOverflow(page);
		const columns = await page
			.locator('.context-menu .item .label')
			.evaluateAll((els) => [...new Set(els.map((e) => Math.round(e.getBoundingClientRect().x)))]);
		expect(columns, 'the labels share one x').toHaveLength(1);
	});

	test('with multi-select on, a plain click adds instead of replacing', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uids: string[] = [];
		// Apart, and to the LEFT: two nodes at one point would cover each other's click target, and the
		// inspector that opens on the first selection takes the right of the canvas.
		for (const [type, cat, x] of [
			['Oscillator', 'inputs', 0],
			['Buffer', 'signal', 260]
		] as const) {
			const uid = await page.evaluate(
				([t, c, px]) => (window as any).goofi.commands.addNode(t, c, [px as number, 0]),
				[type, cat, x] as const
			);
			await page.waitForFunction(
				(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
				uid
			);
			uids.push(uid);
		}
		const selected = () =>
			page.evaluate(() => (window as any).goofi.query.selection().nodes as string[]);
		try {
			// Baseline: without the mode a second plain click REPLACES — the behaviour a phone is stuck
			// with, since it has no shift, ctrl or meta.
			for (const u of uids) await page.locator(`.svelte-flow__node[data-id="${u}"]`).click();
			await expect.poll(selected).toEqual([uids[1]]);

			await openOverflow(page);
			await menuRow(page, 'Multi-select mode').click();
			// From empty: with a node already selected the first click would TOGGLE it back off, which
			// is the same additive semantics shift-click has and not what this case is about.
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			for (const u of uids) await page.locator(`.svelte-flow__node[data-id="${u}"]`).click();
			await expect.poll(async () => (await selected()).slice().sort()).toEqual([...uids].sort());

			// …and a plain click on empty canvas must not wipe it. `clickPane` took the same fold as
			// clickNode/clickEdge: on a phone `shiftKey` is always false, so the biggest target on
			// screen was undoing the very selection the mode exists to build.
			// Found rather than guessed: the pane's corners are taken (the zoom cluster sits in one) and
			// the nodes float wherever the fit put them, so scan for a point that really is bare canvas.
			const empty = await page.evaluate(() => {
				const r = document.querySelector('.svelte-flow__pane')!.getBoundingClientRect();
				for (let y = r.bottom - 20; y > r.top; y -= 20)
					for (let x = r.left + 20; x < r.right; x += 20)
						if (document.elementFromPoint(x, y)?.classList.contains('svelte-flow__pane'))
							return { x, y };
				return null;
			});
			expect(empty, 'the canvas has some bare pane to click').not.toBeNull();
			await page.mouse.click(empty!.x, empty!.y);
			await expect
				.poll(async () => (await selected()).slice().sort(), {
					message: 'a tap on empty canvas leaves the multi-selection alone'
				})
				.toEqual([...uids].sort());
			// …and the CANVAS agrees. SvelteFlow unselects on every pane click, after the callback and
			// whatever the store decided, so the store keeping the selection is only half the fix: the
			// other half is that nothing gated on the rendered flags (Delete, Group, Copy, this row)
			// goes dead while the store still holds the operands.
			await expect(
				page.locator('.svelte-flow__node.selected'),
				'the nodes are still painted selected'
			).toHaveCount(2);

			// So the way out has to be somewhere a pointer can reach — Escape is a keyboard's door only.
			await openOverflow(page);
			await menuRow(page, 'Clear selection').click();
			await expect.poll(selected, { message: 'the menu row is the mode’s deselect' }).toEqual([]);
		} finally {
			await openOverflow(page);
			await menuRow(page, 'Multi-select mode').click();
			await page.evaluate((u) => (window as any).goofi.commands.removeNodes(u), uids);
		}
	});

	/* The hysteresis probe R's audit asked for: where the header's flex line is already overflowing,
	   the status cluster and the tab strip shrink TOGETHER, so two adjacent spill sets can both be
	   self-consistent and the settled bar could depend on which side the width was approached from.
	   The boundary test above only ever approaches from one step away; this walks in from 40px out on
	   each side, at a named patch (the state where the cluster is widest and the band, if there is one,
	   is widest with it). */
	test('one width has one answer, whichever side it is approached from', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await withNamedPatch(page, async () => {
			for (let w = 1000; w >= 360; w -= 20) {
				await widthTo(page, w - 40);
				await widthTo(page, w);
				const fromBelow = (await inBar(page)).join();
				await widthTo(page, w + 40);
				await widthTo(page, w);
				expect((await inBar(page)).join(), `at ${w}px, approached from either side`).toBe(fromBelow);
			}
		});
	});

	/**
	 * Escape belongs to the surface that is dismissing, and to nothing behind it.
	 *
	 * `ContextMenu` claims Escape on a window keydown listener and does not consume it, while
	 * `NodeEditorPanel` registers its own window listener in `onMount` — long before any menu can
	 * mount — so both bubble-phase handlers run and the editor's goes FIRST. Its guards do not
	 * exclude the case: `TopBar` renders outside the panel tree so `activePanelId` never changes, and
	 * the keydown's target is the overflow `<button>`, which is neither in the tag allowlist nor
	 * inside a `dialog[open]`. So backing out of the menu also ran the editor's Escape ladder — the
	 * same defect the file browser's `dialog[open]` guard fixed, on the surface that is the phone's
	 * ONLY door to the canvas commands.
	 *
	 * Both rungs of that ladder are reachable from this exact menu: the rows that need a selection are
	 * `disabled` without one, so a user who came for `Delete selection` demonstrably HAS one; and with
	 * nothing selected the next rung pops a sub-patch level.
	 */
	test('Escape out of the overflow menu leaves the selection alone', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await page.evaluate(() =>
			(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
		);
		await page.waitForFunction(
			(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
			uid
		);
		try {
			await selectNode(page, uid);
			await openOverflow(page);
			// The row a user with a selection came for is live, which is what makes this the ordinary path.
			await expect(menuRow(page, 'Delete selection')).toBeEnabled();

			await page.keyboard.press('Escape');
			await expect(page.locator('.context-menu')).toHaveCount(0);
			expect(
				await page.evaluate(() => (window as any).goofi.query.selection().nodes),
				'dismissing the menu is not also the canvas’s deselect'
			).toEqual([uid]);
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
		}
	});

	test('Escape out of the overflow menu does not pop a sub-patch level', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await page.evaluate(() =>
			(window as any).goofi.commands.addNode('Oscillator', 'inputs', [0, 0])
		);
		await page.waitForFunction(
			(u) => ((window as any).goofi.query.graph().nodes as { uid: string }[]).some((n) => n.uid === u),
			uid
		);
		const inst: string = await page.evaluate(
			(u) => (window as any).goofi.commands.groupNodes([u], [140, 140]),
			uid
		);
		const crumbs = page.getByTestId('subpatch-breadcrumb');
		try {
			await page.locator(`.svelte-flow__node[data-id="${inst}"]`).dblclick();
			await expect(crumbs, 'inside the sub-patch').toBeVisible();
			// The second rung of the ladder needs an EMPTY selection — entering leaves the instance
			// selected, so clear it first.
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.selection().nodes.length))
				.toBe(0);

			await openOverflow(page);
			await page.keyboard.press('Escape');
			await expect(page.locator('.context-menu')).toHaveCount(0);
			await expect(crumbs, 'and the editor is still where it was').toBeVisible();
		} finally {
			if (await crumbs.isVisible())
				await crumbs.getByRole('button', { name: 'Patch', exact: true }).click();
			await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
			await page.evaluate((u) => (window as any).goofi.commands.removeNodes([u]), uid);
		}
	});

	/* ---- The right-bounded identity group (Phil, 2026-08-08) --------------------------------------
	 * The layout tab strip is the header's first-class citizen: it owns the LEFT edge and its content
	 * gets space-priority. The patch identity — perf HUD, filename, dirty dot — moves to the RIGHT and
	 * joins the progressive overflow: it spills into the ⋯ menu BEFORE any action does, as disabled
	 * informational rows. The connection chip stays out of the plan entirely (a warning that overflows
	 * into a hidden menu is not a warning). */

	test('the tab strip owns the left edge; the patch identity sits right, in the overflow group', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const first = page.locator('.topbar > :first-child');
		await expect(first, 'the tab strip is the first section of the bar').toHaveClass(/tabslot/);
		await withNamedPatch(page, async () => {
			const zone = page.locator('.topbar .action-zone');
			await expect(
				zone.locator('[data-testid="topbar-path"]'),
				'the patch name is a resident of the right-hand group'
			).toBeVisible();
			const slot = (await page.locator('.topbar .tabslot').boundingBox())!;
			const p = (await zone.locator('[data-testid="topbar-path"]').boundingBox())!;
			expect(p.x, 'name to the right of the strip').toBeGreaterThan(slot.x + slot.width - 1);
		});
	});

	test('the identity yields to the menu before any action does', async ({ page }) => {
		// 430px: too narrow for the (32ch-capped) name AND the actions, wide enough for the actions
		// alone — the exact band where the priority order is observable.
		await page.setViewportSize({ width: 430, height: 720 });
		await page.goto('/');
		await waitForApp(page);
		await withNamedPatch(page, async () => {
			await expect(page.locator('.topbar [data-testid="topbar-path"]')).toBeHidden();
			const kept = await settledBar(page);
			// Set-compare: `inBar` reads DOM order (undo·redo·save·caret·load), PRIORITY is spill
			// order — same members, different sequences.
			expect(new Set(kept), 'every action outranks the identity').toEqual(new Set(PRIORITY));
			await openOverflow(page);
			try {
				const row = page.locator('.context-menu .item', { hasText: CROWDING_NAME.slice(0, 24) });
				await expect(row, 'the name is a row in the menu instead').toHaveCount(1);
				await expect(row, 'informational, not clickable').toBeDisabled();
			} finally {
				await page.keyboard.press('Escape');
			}
		});
	});

	test('layout tabs get the room the bar gives up', async ({ page }) => {
		await page.setViewportSize({ width: 800, height: 720 });
		await page.goto('/');
		await waitForApp(page);
		await withNamedPatch(page, async () => {
			// At 800px the name chip fits comfortably beside every action…
			await expect(page.getByTestId('topbar-path')).toBeVisible();
			// …then SIX more layout tabs claim the width. Six is measured, not guessed: at 800px the
			// full row (chips + actions + trigger + gaps) needs ~370px against a ~772px budget-before-
			// reserve, and each "Layout N" pill costs ~60px on the ~87px single-tab strip — so the
			// identity's spill boundary sits at seven tabs, with ~50px of margin against the rem
			// clamp. (Four and five both landed inside the noise, which is how this comment earned
			// its arithmetic.) The reserve follows the strip's CONTENT, so the identity yields into
			// the menu in the tabs' favour — and the strip fits whole instead of scrolling while
			// chips sit comfortable.
			for (let i = 0; i < 6; i++) {
				await page.getByRole('button', { name: 'New tab' }).click();
			}
			try {
				await expect(
					page.getByTestId('topbar-path'),
					'the identity handed its width to the tabs'
				).toBeHidden();
				await expect
					.poll(async () => {
						const m = await page
							.locator('[data-testid="workspace-tabs"]')
							.evaluate((el) => ({ content: el.scrollWidth, box: el.clientWidth }));
						return m.content <= m.box + 1;
					}, 'the strip fits its content — nothing is scrolled away')
					.toBe(true);
				// The actions keep their slots at this width — the room came out of the identity first,
				// which is the declared order doing its job.
				expect(new Set(await settledBar(page))).toEqual(new Set(PRIORITY));
			} finally {
				// The six tabs are this test's own to hand back. It used to get them collapsed for free,
				// because the old reset was a LOAD of the patch saved before they existed. `resetPatch`
				// resets the PATCH, and the arrangement is not part of it: a `new` snapshot carries no
				// layout, so the client keeps what is on screen — and pushes it into the fresh patch.
				const close = page.getByRole('button', { name: 'Close tab' });
				for (let i = 0; i < 6; i++) await close.last().click();
				await expect(page.getByTestId('workspace-tabs').locator('.ui-tab')).toHaveCount(1);
			}
		});
	});
});

test.describe('the connection status', () => {
	/**
	 * What the header says about the control connection — which, while it is healthy, is nothing.
	 *
	 * Phil's call: "the 'connected' state should not indicate anything, we don't need to communicate
	 * 'everything is good'. Let's only communicate when something needs attention." So the badge that
	 * used to sit in the status cluster at every moment of the app's life — restating the one fact that
	 * is true in every screenshot of a working app — is gone, and the width it spent goes to the
	 * filename and the tab strip that were fighting over it.
	 *
	 * The alarm it becomes is deliberately louder than the badge it replaces: the badge alone is 72px
	 * of a 412px bar and easy to miss, so the whole header wears a thick warning outline as well.
	 */

	/** Record the app's control socket, and give the test a way to cut it the way a backend going away
	 * does — a real `close` through the client's own handler, not a store poke.
	 *
	 * The reconnect is frozen deliberately. `api/control.ts` retries on a 250ms→5s backoff, so a bare
	 * close would flicker (drop → reconnect → drop) under a running backend and every assertion below
	 * would be racing it. Replacing the constructor with one that never fires `open` OR `close` leaves
	 * the client exactly where a real unreachable backend leaves it: disconnected, one dangling attempt
	 * outstanding, nothing scheduled. */
	async function armControlCut(page: import('@playwright/test').Page): Promise<void> {
		await page.addInitScript(() => {
			const Native = window.WebSocket;
			const control: WebSocket[] = [];
			class Recorded extends Native {
				constructor(url: string | URL, protocols?: string | string[]) {
					super(url, protocols);
					if (String(url).includes('/control')) control.push(this);
				}
			}
			window.WebSocket = Recorded as unknown as typeof WebSocket;
			(window as any).__cutControl = () => {
				window.WebSocket = class {
					readyState = 0;
					addEventListener() {}
					removeEventListener() {}
					send() {}
					close() {}
				} as unknown as typeof WebSocket;
				for (const ws of control) ws.close();
			};
		});
	}

	test('says nothing at all while the connection is healthy', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await expect(page.getByTestId('topbar-connection'), 'no chip, and no width spent').toHaveCount(0);
		// Read off `textContent`, like `topbar.spec.ts` reads the brand: the claim is that the header
		// carries no connection state at all, not merely that it is painted out of sight.
		const text = ((await page.locator('.topbar').textContent()) ?? '').toLowerCase();
		expect(text, 'a healthy socket is not news').not.toContain('connect');
		await expect(page.locator('.topbar'), 'and the bar wears no alarm').toHaveCSS(
			'outline-style',
			'none'
		);
	});

	test('a LOST connection takes space in the bar and frames the whole window', async ({ page }) => {
		await armControlCut(page);
		await page.goto('/');
		await waitForApp(page);
		const bar = page.locator('.topbar');
		const before = await bar.boundingBox();

		await page.evaluate(() => (window as any).__cutControl());

		// (a) the chip, in the bar — kept out of the progressive overflow, so a warning can never
		// spill into a menu the user has to open to find it.
		const chip = page.locator('.topbar [data-testid="topbar-connection"]');
		await expect(chip).toHaveText(/disconnected/i);
		await expect(
			page.locator('.topbar .actions [data-testid="topbar-connection"]'),
			'it is not one of the spillable actions'
		).toHaveCount(0);

		// (b) the whole WINDOW, framed in the warning ink — the bar-only outline read as the bar's
		// problem; a lost backend is the app's. The frame is a fixed, pointer-transparent overlay so
		// it can sit above every panel without stealing a single event.
		const frame = page.getByTestId('net-frame');
		await expect(frame).toBeVisible();
		await expect(frame, 'the frame spans the viewport').toHaveCSS('position', 'fixed');
		await expect(frame).toHaveCSS('pointer-events', 'none');
		await expect(frame, 'a thick warning ring, drawn inward').toHaveCSS(
			'box-shadow',
			/rgb\(240, 192, 80\).*inset|inset.*rgb\(240, 192, 80\)/ // --warning
		);
		const fb = await frame.boundingBox();
		const vp = page.viewportSize()!;
		expect(fb, 'the frame covers the whole window, not one bar').toEqual({
			x: 0,
			y: 0,
			width: vp.width,
			height: vp.height
		});

		// …and nothing moved: the frame is an overlay, so the bar keeps its box and the workspace
		// below it keeps its origin.
		await expect(bar, 'the bar itself wears no outline any more').toHaveCSS('outline-style', 'none');
		expect(await bar.boundingBox(), 'the alarm costs no layout').toEqual(before);
	});
});

test.describe('the icon set', () => {
	/**
	 * ONE ICON SYSTEM — the app draws every icon itself, and lets the browser draw none.
	 *
	 * The defect this pins out was that "an icon" meant two different things depending on where you
	 * looked. Most were Unicode glyphs (`✕`, `⋯`, `↶`, `▾`) resolved by the OS font stack, so the app
	 * looked different on every platform and had no say in the weight or the shape. The rest were not
	 * the app's at all: `MetadataPanel` rendered a bare `<summary>`, whose disclosure marker is drawn
	 * by the BROWSER — a triangle in Chrome, a different one in Firefox, in a colour the palette never
	 * chose.
	 *
	 * Both are replaced by one renderer over vendored Lucide geometry (`$lib/ui/Icon`), which is what
	 * the `svg[data-icon]` assertions below read. `icons.test.ts` guards the geometry itself; this file
	 * guards the thing only the running app can answer — that the browser is no longer drawing.
	 *
	 * Runs under the `default` (fine-pointer) project; the icons are pointer-independent.
	 */

	const CHROME_GLYPHS = ['✕', '×', '⋯', '↶', '↷', '▾', '▸', '▶', '＋', '▕', '▁', '⤢', '⟳'];

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

	/* The one place the BROWSER was still drawing. `<details>`/`<summary>` is the right element for a
	   collapsible meta field and stays — what goes is its UA marker, replaced by the app's own chevron
	   so the affordance is the app's shape in the app's ink. */
	test('the metadata tree draws its own chevron, and no native <details> marker', async ({ page }) => {
		const uid = await selectedOscillator(page);
		try {
			const field = page.getByTestId('auto-side-panel').locator('.meta-field').first();
			// The panel renders off the live data stream, so the first frame has to land first.
			await expect(field, 'a running Oscillator puts at least one meta field on screen').toBeVisible();
			const summary = field.locator('summary');

			await expect(
				summary,
				'no UA disclosure marker — the browser draws nothing here'
			).toHaveCSS('list-style-type', 'none');
			await expect(
				summary.locator('svg[data-icon]'),
				'the affordance is the app’s own chevron instead'
			).toHaveCount(1);

			// And it is a real affordance, not decoration: it turns with the section it marks.
			await expect(summary.locator('svg[data-icon]')).toHaveAttribute('data-icon', 'chevron-right');
		} finally {
			await drop(page, uid);
		}
	});

	/* The app header, the tab strip and the panel header are the three resident chrome bars — between
	   them they carried nine of the Unicode glyphs. Asserted as a property of the RENDERED text, so a
	   glyph that comes back anywhere in those bars fails here rather than at the site that reintroduced
	   it. */
	test('the resident chrome bars render icons, never a text glyph', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const bars = page.locator('.topbar, [data-testid="panel-header"]');
		await expect(bars, 'the app header and the panel header are both on screen').toHaveCount(2);
		const text = (await bars.allTextContents()).join('');
		for (const g of CHROME_GLYPHS)
			expect(text, `the chrome bars no longer render "${g}" as text`).not.toContain(g);

		// The controls that carried those glyphs each name the icon they draw now.
		for (const [testid, icon] of [
			['topbar-undo', 'undo-2'],
			['topbar-redo', 'redo-2'],
			['topbar-save-caret', 'chevron-down'],
			['topbar-overflow', 'ellipsis'],
			['panel-split-row', 'square-split-horizontal'],
			['panel-split-column', 'square-split-vertical'],
			['panel-maximize', 'maximize-2'],
			['panel-close', 'x']
		] as const) {
			await expect(
				page.getByTestId(testid).locator('svg'),
				`${testid} draws the ${icon} icon`
			).toHaveAttribute('data-icon', icon);
		}
	});

	/* Phil's call, and the reason it is worth a line of its own: the two splits are Lucide's two
	   DRAWN icons, not one icon rotated 90°. A rotation would read as the same control twice, and it
	   is what the header's own naming ("a vertical divider ▕ / a horizontal divider ▁") used to say
	   in a comment because the glyphs could not say it themselves. */
	test('the two panel splits are two distinct icons, neither of them rotated', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const right = page.getByTestId('panel-split-row').locator('svg');
		const down = page.getByTestId('panel-split-column').locator('svg');

		await expect(right).toHaveAttribute('data-icon', 'square-split-horizontal');
		await expect(down).toHaveAttribute('data-icon', 'square-split-vertical');
		await expect(right, 'drawn as-is, not turned').toHaveCSS('transform', 'none');
		await expect(down, 'drawn as-is, not turned').toHaveCSS('transform', 'none');

		// The mapping is unchanged: Split Right still makes a row, Split Down still makes a column.
		await expect(page.getByTestId('panel-split-row')).toHaveAttribute('title', 'Split Right');
		await expect(page.getByTestId('panel-split-column')).toHaveAttribute('title', 'Split Down');
	});
});

test.describe('the focus ring', () => {
	// A keyboard-driven tool must show focus. Before F there were zero :focus-visible rules.
	// Drive an actual keyboard Tab — programmatic .focus() does not reliably trigger :focus-visible,
	// but keyboard focus does, and our universal `:focus-visible` rule rings whatever gets focused.
	test('a keyboard-focused element shows the app accent focus ring', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page); // deterministic hydration (same gate the other specs use)
		await page.keyboard.press('Tab');
		const ring = await page.evaluate(() => {
			const el = document.activeElement as HTMLElement | null;
			if (!el || el === document.body) return { focused: false, outlineWidth: '', outlineColor: '' };
			const s = getComputedStyle(el);
			return { focused: true, outlineWidth: s.outlineWidth, outlineColor: s.outlineColor };
		});
		expect(ring.focused, 'Tab must move focus into the app').toBe(true);
		// Assert the app rule specifically, not merely "some outline" — a future UA-default outline would
		// pass a bare `!== 'none'` as a tautology. The rule is `2px solid var(--accent)`, #50d0a0.
		expect(ring.outlineWidth, 'the app :focus-visible ring is 2px').toBe('2px');
		expect(ring.outlineColor, 'the ring colour is --accent (#50d0a0)').toBe('rgb(80, 208, 160)');
	});

	// ...and shows it exactly ONCE. app.css's `input:focus { border-color: var(--accent) }` (0,0,1) does
	// not beat `:focus-visible` (0,1,0), so a focused field used to paint the accent twice: a 1px accent
	// border, a 1px gap, then the 2px accent outline. Against the "too salient" brief that fires the
	// loudest ink in the palette — the same ink as node selection and the active-panel ring — twice for
	// one state. Only a computed readback can tell the two rings apart.
	test('a focused text field paints ONE accent ring, not two concentric ones', async ({ page }) => {
		await page.goto('/dev/ui');
		const input = page.getByTestId('ui-field-number');
		await input.waitFor();
		await input.click();

		const ring = await input.evaluate((el) => {
			const s = getComputedStyle(el);
			return {
				focused: document.activeElement === el,
				outlineWidth: s.outlineWidth,
				outlineColor: s.outlineColor,
				borderColor: s.borderTopColor
			};
		});
		expect(ring.focused, 'clicking the field focuses it').toBe(true);
		// Browsers match :focus-visible on a text input even under mouse focus (a UA heuristic), so the
		// outline half is the real, rendered ring — not a proxy for it.
		expect(ring.outlineWidth, 'the single ring is the 2px :focus-visible outline').toBe('2px');
		expect(ring.outlineColor, 'and it is --accent').toBe('rgb(80, 208, 160)');
		expect(ring.borderColor, 'the field keeps its resting --border hairline (no second ring)').toBe(
			'rgb(72, 72, 72)'
		);
	});
});

test.describe('the user-agent reset', () => {
	// app.css resets `button { font: inherit }` and `input, select, textarea { font: inherit }` for one
	// documented reason: a UA `font` DECLARATION beats inheritance, so those elements would otherwise
	// fall out of the app face entirely. `code`/`pre`/`kbd`/`samp` are the same class of element and had
	// no such rule — they rendered in the browser's generic monospace, whatever the page around them
	// said.
	//
	// What the reset buys is NOT a face: it is that the face is decided by the cascade rather than by
	// the UA. Under the two-face taxonomy that distinction is what these two tests pull apart — the
	// same reset hands a CHROME `<code>` the sans it inherits and a DATA `<pre>` the mono its component
	// declares, and neither one gets the browser's generic monospace.

	// The probe is the one `<code>` in the tree that declares no font of its own: Panel's unknown-type
	// fallback, which a `.gfi` layout naming a retired panel reaches for real. It is chrome (an
	// explanatory message in a panel body), so with no UA default leaking it renders the body sans.
	//
	// It is reached the way production reaches it — a STORED layout naming a type this build does not
	// have — because that is now the only way there is. `page_set_panel` refuses a panel type outside
	// the manager's vocabulary, so the shortcut this spec used to take (setPanelType with a made-up
	// name) is exactly the mistake the manager exists to catch. A load still admits one: the fallback's
	// whole reason to exist is a patch saved by a build that had the type.
	async function loadRetiredPanelType(page: import('@playwright/test').Page): Promise<void> {
		await page.evaluate(async () => {
			const ws = new WebSocket(`ws://${location.host}/control`);
			await new Promise((r) => (ws.onopen = r));
			const call = (op: string, payload: unknown): Promise<any> =>
				new Promise((res) => {
					const id = Math.floor(Math.random() * 1e6);
					const on = (e: MessageEvent) => {
						if (typeof e.data !== 'string') return;
						const m = JSON.parse(e.data);
						if (m.id !== id) return;
						ws.removeEventListener('message', on);
						res(m);
					};
					ws.addEventListener('message', on);
					ws.send(JSON.stringify({ id, op, payload }));
				});
			const yaml: string = (await call('serialize', {})).result.yaml;
			await call('load_text', {
				content: yaml.replace(/panel_type: node-editor/g, 'panel_type: retired-panel-type')
			});
			ws.close();
		});
	}

	test('a <code> with no rule of its own inherits the chrome face, not the UA monospace', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		try {
			await loadRetiredPanelType(page);
			const code = page.locator('.panel .missing code');
			await expect(code).toBeVisible();
			const font = await code.evaluate((el) => getComputedStyle(el).fontFamily);
			expect(font, 'the UA reset hands <code> back to the inherited app face').toContain('Inter');
		} finally {
			// Never leave a bogus panel type in the manager's stored layout for the next spec.
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		}
	});

	// The reset is a RESET, not a skin: `code, pre, kbd, samp` scores (0,0,1), so every component rule
	// that sizes one of these elements still wins — the family comes back by inheritance, the RUNG does
	// not come from here. The console row is the reachable one, and it is the DATA half of the pair
	// above: the mono it renders is the one its own panel declares, arriving through `font: inherit`.
	// (The restatements this reset once made look redundant are load-bearing again under two faces —
	// a data surface that states no family follows `body` into the chrome face.)
	test('the reset hands back family only — a component rule still owns the size', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

		// A node erroring on its empty required input (see `addErroringNode`); the graph store mirrors
		// the error into the console as a stderr line. The cheapest real console content.
		const uid = await addErroringNode(page);
		try {
			const txt = page.getByTestId('console-entry').first().locator('.txt');
			await expect(txt, 'the node error reached the console').toBeVisible();

			const s = await txt.evaluate((el) => ({
				family: getComputedStyle(el).fontFamily,
				size: parseFloat(getComputedStyle(el).fontSize),
				rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
			}));
			expect(s.family, 'the <pre> row inherits the app mono face').toContain('JetBrains Mono');
			expect(s.size, 'and keeps its own --fs-small rung, not the UA smaller-monospace').toBeCloseTo(
				0.82 * s.rem,
				0
			);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		}
	});
});

test.describe('the device stamp', () => {
	// The fake keyboard lives in `lib/` because `touch-expr.spec.ts` needs the same one (X's completion
	// popup clamps against the visual viewport too), and a second copy is a second thing to keep true.

	/**
	 * The device seam's ONE surviving output (D-R8): `--kb-inset`, how far the soft keyboard overlaps
	 * the layout viewport. `data-pointer` / `data-size` / `data-short` were deleted with `classify()` —
	 * they were write-only, and `@media` already answers the question they encoded. `--kb-inset` is
	 * kept because nothing in CSS can answer it: the overlap is observable only through
	 * `visualViewport`.
	 *
	 * So this spec asserts BEHAVIOUR, not the presence of an attribute: the value tracks the visual
	 * viewport, and the two anchored-overlay clamps place against the visual viewport rather than the
	 * layout one — a menu opened with the keyboard up must not land underneath it.
	 */

	const MARGIN = 6; // clampToViewport's viewport-edge margin

	test('--kb-inset tracks the visual viewport, so the soft keyboard is measurable', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		expect(await kbInset(page), 'no keyboard ⇒ no inset').toBe('0px');

		await setKeyboardInset(page, 300);
		await expect.poll(() => kbInset(page), { message: 'the inset follows visualViewport' }).toBe(
			'300px'
		);

		await setKeyboardInset(page, 0);
		await expect.poll(() => kbInset(page), { message: 'and returns to 0 when it closes' }).toBe(
			'0px'
		);
	});

	/**
	 * Both clamps read `window.innerHeight` — the LAYOUT viewport, which the keyboard does not shrink.
	 * The inset is derived from the overlay's own measured box so the test tunes itself: enough to push
	 * its bottom edge under the keyboard, not so much that it no longer fits above it.
	 */
	async function expectsClampsAboveTheKeyboard(
		page: Page,
		openMenu: () => Promise<void>,
		menu: Locator
	): Promise<void> {
		await openMenu();
		await expect(menu).toBeVisible();
		const before = (await menu.boundingBox())!;
		const innerHeight = await page.evaluate(() => window.innerHeight);
		await page.keyboard.press('Escape');
		await expect(menu).toBeHidden();

		const inset = innerHeight - (before.y + before.height) + 20;
		expect(inset, 'the overlay starts clear of the bottom edge, so an inset can reach it').toBeGreaterThan(0);

		await setKeyboardInset(page, inset);
		try {
			await openMenu();
			await expect(menu).toBeVisible();
			const box = (await menu.boundingBox())!;
			const visualHeight = innerHeight - inset;
			expect(box.y, 'the overlay stays on-screen').toBeGreaterThanOrEqual(MARGIN - 0.5);
			expect(
				box.y + box.height,
				'the overlay sits above the keyboard, not underneath it'
			).toBeLessThanOrEqual(visualHeight - MARGIN + 0.5);
			await page.keyboard.press('Escape');
			await expect(menu).toBeHidden();
		} finally {
			await setKeyboardInset(page, 0);
		}
	}

	test('ContextMenu places against the visual viewport, not under the keyboard', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await expectsClampsAboveTheKeyboard(
			page,
			() => page.getByTestId('topbar-save-caret').click(),
			page.locator('.context-menu').first()
		);
	});

	test('Popover places against the visual viewport, not under the keyboard', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		const slot = page.locator(`.slot-viewer[data-node="${uid}"]`);
		try {
			await expectsClampsAboveTheKeyboard(
				page,
				() => slot.getByTestId('viewer-settings-cog').click(),
				page.getByTestId('viewer-settings-menu')
			);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});

	/**
	 * The other inset the device seam owns, and the one F shipped as a no-op: the notch / rounded-corner
	 * / home-indicator safe area. `viewport-fit=cover` (app.html) makes the app draw under all three, so
	 * the padding is what keeps the TopBar and the bottom-edge controls out from under them.
	 *
	 * It was stated on `body`. The shell is `position: fixed; inset: 0`, so it is laid out against the
	 * INITIAL CONTAINING BLOCK — nothing on `body` can move it, and phase 1 below pins exactly that, so
	 * the rule cannot drift back onto an ancestor that does not contain the app. Chromium's device
	 * emulation reports `env()` as 0, which is why all four projects were green against a dead rule; the
	 * insets are therefore named as tokens and stamped here, which is also how a surface that has to
	 * restate them (`Toast`, itself fixed) stays in step with the shell.
	 */
	test('the safe-area inset is stated where the app chrome can actually feel it', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const bar = page.locator('.topbar');
		const panel = page.locator('.panel').first();
		const before = (await bar.boundingBox())!;
		const panelBefore = (await panel.boundingBox())!;

		// 1. Padding on `body` cannot reach a fixed shell — the defect, made permanent as a guard.
		await page.addStyleTag({ content: 'body { padding: 44px 20px 34px 20px !important; }' });
		const onBody = (await bar.boundingBox())!;
		expect(onBody.y, 'a fixed shell is laid out against the viewport, not against body').toBe(before.y);
		expect(onBody.x, 'in both axes').toBe(before.x);

		// 2. Stated on the shell itself, every edge of the app chrome moves off the unsafe area.
		await page.evaluate(() => {
			const s = document.documentElement.style;
			s.setProperty('--safe-top', '44px');
			s.setProperty('--safe-right', '20px');
			s.setProperty('--safe-bottom', '34px');
			s.setProperty('--safe-left', '20px');
		});
		const after = (await bar.boundingBox())!;
		const panelAfter = (await panel.boundingBox())!;
		expect(after.y - before.y, 'the top bar clears the notch').toBe(44);
		expect(after.x - before.x, 'and the rounded left edge').toBe(20);
		expect(before.width - after.width, 'the bar gives up both side insets').toBe(40);
		expect(
			panelBefore.y + panelBefore.height - (panelAfter.y + panelAfter.height),
			'and the bottom-edge controls clear the home indicator'
		).toBe(34);
	});
});
