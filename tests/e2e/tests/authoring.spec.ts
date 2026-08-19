// Building a patch on the canvas: adding, naming, wiring, undoing, and stepping into a
// sub-patch. One describe per situation.

import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import {
	bareSpot,
	clearStage,
	inConn,
	inputLabel,
	outHandle,
	proximityStage
} from '../lib/cableDrag';
import { controlsInset } from '../lib/editor';
import {
	addErroringNode,
	addNode,
	nodeParams,
	nodes,
	redo,
	selectNode,
	undo,
	waitForNoNode,
	waitForNode
} from '../lib/goofi';
import { BAR_PANELS } from '../lib/panelBar';
import { paletteItem } from '../lib/placement';
import { emptySpot } from '../lib/touch';

test.describe('adding a node from the picker', () => {
	/**
	 * The second door onto a panel's node binding (the first is a drag from an editor, which needs an
	 * editor on the same page and a pointer that can drag between two of them — neither of which a
	 * phone, or a layout page holding only viewers, has).
	 *
	 * One control, `$lib/panels/NodeSelect`, worn identically by every panel type the manager marks
	 * `acceptsNode`; these tests drive it through the same testid in each, which is what "in the same
	 * way" has to mean to be checkable.
	 */

	const panelId = (page: Page): Promise<string> =>
		page.evaluate(() => (window as any).goofi.query.panels()[0].panelId);

	const panelView = (page: Page, id: string): Promise<{ node: string | null; slot: string | null }> =>
		page.evaluate(
			(pid) => (window as any).goofi.query.panels().find((p: any) => p.panelId === pid),
			id
		);

	/** Borrow the sole panel as `type`, and hand it back to the editor afterwards. */
	async function borrow(page: Page, type: string, ready: string): Promise<() => Promise<void>> {
		const id = await panelId(page);
		await page.evaluate(
			([p, t]) => (window as any).goofi.commands.setPanelType(p, t),
			[id, type] as const
		);
		await expect(page.getByTestId(ready)).toBeVisible();
		return async () => {
			await page.evaluate(
				(p) => (window as any).goofi.commands.setPanelType(p, 'node-editor'),
				id
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
		};
	}

	for (const [type, ready] of BAR_PANELS) {
		test(`the ${type} panel binds a node from its bar, with nothing dragged`, async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			const id = await panelId(page);
			const restore = await borrow(page, type, ready);
			try {
				const picker = page.getByTestId('panel-node').locator('select');
				// Present BEFORE anything is bound — an unbound panel that hides its own picker leaves a
				// phone no way in at all, which is the defect this door exists to close.
				await expect(picker, 'the picker is there with nothing bound').toBeVisible();
				expect((await panelView(page, id)).node, 'and nothing is bound yet').toBeNull();

				await picker.selectOption(uid);
				await expect
					.poll(async () => (await panelView(page, id)).node, { timeout: 5_000 })
					.toBe(uid);
				// The option reads as the node's DISPLAY name while the committed value is its uid.
				expect(await picker.locator('option:checked').innerText()).toBe('oscillator0');

				await picker.selectOption('');
				await expect.poll(async () => (await panelView(page, id)).node).toBeNull();
			} finally {
				await restore();
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid);
			}
		});
	}

	test('a binding whose node was removed reads as unbound, never as a raw uid', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		const id = await panelId(page);
		const restore = await borrow(page, 'metadata', 'node-linked-panel');
		try {
			const picker = page.getByTestId('panel-node').locator('select');
			await picker.selectOption(uid);
			await expect.poll(async () => (await panelView(page, id)).node).toBe(uid);

			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);

			await expect
				.poll(async () => picker.inputValue(), { timeout: 5_000 })
				.toBe('');
			expect(
				await picker.locator('option').allInnerTexts(),
				'the dead node is gone from the list too'
			).toEqual(['No node']);
		} finally {
			await restore();
		}
	});

	/* The viewer binds a node AND a slot, so a pick that only settled half of it would leave the panel
	   on a slot name the new node has never had. `linkNodeToPanel` clears the slot in the same merged
	   write (pinned exactly, by payload, in `workspace.svelte.test.ts`); what THIS measures is the
	   consequence — a viewer picked from the bar is drawing, with no second gesture. */
	test('a viewer picked from the bar is fully bound, slot and all', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);
		const id = await panelId(page);
		const restore = await borrow(page, 'viewer', 'node-linked-panel');
		try {
			await page.getByTestId('panel-node').locator('select').selectOption(osc);
			await expect.poll(async () => (await panelView(page, id)).node).toBe(osc);

			const slot = page.getByTestId('viewer-slot').locator('select');
			await expect(slot, 'the slot picker came up with the node').toBeVisible();
			expect(await slot.inputValue(), 'settled on a real output slot, not on nothing').not.toBe('');
			await expect(page.getByTestId('viewer-kind'), 'and the viewer itself is drawing').toBeVisible();
		} finally {
			await restore();
			await page.evaluate((x) => (window as any).goofi.commands.removeNode(x), osc);
			await waitForNoNode(page, osc);
		}
	});
});

test.describe('choosing a slot when a cable has more than one target', () => {
	/**
	 * The product-layer guard for M's one known user-visible regression: a sub-patch's renamed port
	 * names vanished from the Viewer and Metadata slot pickers, because the `Select` migration dropped
	 * the option-text/option-value split those two panels depend on. `5f445e0` fixed it with
	 * `Select.labels` — but the guard went into the ui-gallery, which proves only that the PRIMITIVE
	 * renders a labels map it is handed. Deleting `labels={…}` from either panel left the whole suite
	 * green, so the exact regression could re-ship.
	 *
	 * The split is not cosmetic. A portal's boundary id (`out0`) is the routing key external wires and
	 * every read resolve through; its NAME is a free-text label a rename changes without re-keying
	 * anything. A picker that commits the label breaks the wire; one that shows the id shows the user
	 * an implementation detail instead of the port they named.
	 */
	test('the slot pickers show a sub-patch port’s name and commit its boundary id', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);

		// A sub-patch whose one exposed output is deliberately renamed, so the label and the routing
		// key differ — the whole point. Grouping alone would not do: a fresh port's name DEFAULTS to
		// its id, which is precisely the case a broken picker still passes.
		const inst: string = await page.evaluate(
			(u) => (window as any).goofi.commands.groupNodes([u], [40, 40]),
			osc
		);
		// A collapsed instance is SYNTHESIZED per scene, so it never appears in the flat `graph().nodes`
		// list — `query.node`/`query.instance` are what resolve it.
		await page.waitForFunction((i) => (window as any).goofi.query.instance(i) !== null, inst);
		const bnd: string = await page.evaluate(
			(i) => (window as any).goofi.commands.addBoundary(i, 'out', 'ARRAY', [200, 40]),
			inst
		);
		await page.evaluate(
			([i, b, u]) => (window as any).goofi.commands.wireBoundary(i, b, u, 'out'),
			[inst, bnd, osc] as const
		);
		await page.evaluate(
			([i, b]) => (window as any).goofi.commands.renameBoundary(i, b, 'envelope'),
			[inst, bnd] as const
		);
		await page.waitForFunction(
			([i, b]) => (window as any).goofi.query.node(i)?.slot_labels?.[b] === 'envelope',
			[inst, bnd] as const
		);

		// Borrow the default editor panel as each picker's host in turn, and give it back after.
		// Order matters: switching a panel's type discards the old type's state, the binding included.
		const host = async (type: string) => {
			await page.evaluate(
				([id, t]) => (window as any).goofi.commands.setPanelType(id, t),
				[panelId, type] as const
			);
			await page.evaluate(
				([id, u]) => (window as any).goofi.commands.bindNodeToPanel(id, u),
				[panelId, inst] as const
			);
		};

		try {
			await host('viewer');
			const viewerSlot = page.getByTestId('viewer-slot').locator('select');
			await expect(viewerSlot.locator('option')).toHaveText([`envelope · array`]);
			await viewerSlot.selectOption({ label: 'envelope · array' });
			await expect(viewerSlot, 'the viewer picker commits the routing key').toHaveValue(bnd);

			await host('metadata');
			const metaSlot = page.getByTestId('metadata-slot').locator('select');
			await expect(metaSlot.locator('option')).toHaveText(['envelope']);
			await metaSlot.selectOption({ label: 'envelope' });
			await expect(metaSlot, 'the metadata picker commits the routing key').toHaveValue(bnd);
		} finally {
			await page.evaluate(
				(id) => (window as any).goofi.commands.setPanelType(id, 'node-editor'),
				panelId
			);
			await expect(page.locator('.canvas-wrap').first(), 'the editor panel is back').toBeVisible();
			// RemoveNode captures the whole subtree, so deleting the collapsed instance takes its member
			// and stubs with it.
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), inst);
			await page
				.waitForFunction((u) => (window as any).goofi.query.node(u) === null, osc)
				.catch(() => {});
			// The restore is a command against the RUNNING PATCH, which outlives this page — and the
			// assertion above only passes once the manager's delta drew it back, so it has landed.
		}
	});
});

test.describe('a slot names itself', () => {
	/**
	 * The fine-pointer half of `touch-slot-name.spec.ts`.
	 *
	 * The proximity reveal is ADDITIVE here: a mouse keeps the hover reveal it always had (and the
	 * `:focus-visible` one beside it), and gains the same in-flight proximity reveal touch now depends
	 * on. Both are asserted, because "the existing specs still pass" cannot prove the first — nothing
	 * else in the suite reads this label's opacity on a mouse.
	 */

	test('hover still reveals an input’s name, and a cable in flight reveals it too', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const stage = await proximityStage(page);
		try {
			const label = inputLabel(page, stage.near);
			await expect(label, 'hidden at rest, as it always was on a mouse').toHaveCSS('opacity', '0');

			// The reveal the mouse already had, and must keep.
			const pill = await inConn(page, stage.near);
			await page.mouse.move(pill.x, pill.y);
			await expect(label, 'hovering the connector still names it').toHaveCSS('opacity', '1');
			await page.mouse.move(2, 2);
			await expect(label).toHaveCSS('opacity', '0');

			// …and the one it gains: while a cable is in flight, proximity alone is enough — the pointer
			// stops short of the pill, so nothing here can be the hover rule answering in disguise.
			const from = await outHandle(page, stage.src);
			await page.mouse.move(from.x, from.y);
			await page.mouse.down();
			try {
				for (const f of [0.4, 0.75, 1]) {
					await page.mouse.move(
						Math.round(from.x + (pill.x - 20 - from.x) * f),
						Math.round(from.y + (pill.y - from.y) * f)
					);
					await page.waitForTimeout(30);
				}
				await expect(
					label,
					'the input the cable is closing on names itself, 20px short of the pill'
				).toHaveCSS('opacity', '1');
				await expect(
					inputLabel(page, stage.far),
					'…and a distant one stays quiet'
				).toHaveCSS('opacity', '0');
			} finally {
				// Release over bare canvas, so this measures the reveal and never wires anything.
				const away = await bareSpot(page);
				await page.mouse.move(away.x, away.y);
				await page.mouse.up();
			}
			await expect(label, 'the tag goes away with the cable').toHaveCSS('opacity', '0');
		} finally {
			await clearStage(page, stage);
		}
	});
});

test.describe('renaming a node', () => {
	/**
	 * `control.ts` states the split outright: the uid is the universal identity everything references
	 * the node by, and the name is "for the label only". Two labels broke the second half by rendering
	 * the identity — the add-node menu's seed chip (`from 000000000003.out`, on the primary
	 * add-and-auto-wire flow) and the console's per-row source button, which sat beside canvas nodes
	 * named `oscillator0`. Both are the one place their surface names the node at all, and neither was
	 * pinned, so a resolution that regresses would ship silently.
	 *
	 * Asserted as "shows the name and NOT the 12-hex uid", because the uid legitimately stays in the
	 * DOM around both (`data-node`, the click's focus/autoLink routing) — only the visible text moved.
	 */
	const HEX_UID = /^[0-9a-f]{12}$/;

	test('the add-menu seed chip names the node it will wire from', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const uid = await addNode(page, 'Oscillator', 'inputs', [60, 60]);
		await waitForNode(page, uid);
		expect(uid, 'the identity really is an opaque hex string').toMatch(HEX_UID);
		const name: string = await page.evaluate((u) => (window as any).goofi.query.node(u).name, uid);

		try {
			// The connector pill click — the flow that opens the menu seeded from a slot.
			await page
				.locator(`.svelte-flow__node[data-id="${uid}"]`)
				.getByTestId('slot-output-pin')
				.first()
				.click();
			const chip = page.getByTestId('add-menu-seed');
			await expect(chip, 'the menu opened seeded from that slot').toBeVisible();
			await expect(chip.locator('.seed-ref')).toHaveText(`${name}.out`);
		} finally {
			await page.keyboard.press('Escape');
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});

	test('a console row names its source node', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);

		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		// Borrow the default editor panel and give it straight back (see console-rows.spec.ts).
		await page.evaluate((id) => (window as any).goofi.commands.setPanelType(id, 'console'), panelId);

		// A node erroring on its empty required input (see `addErroringNode`) — the cheapest real
		// console content.
		const uid = await addErroringNode(page);
		try {
			const name: string = await page.evaluate((u) => (window as any).goofi.query.node(u).name, uid);
			expect(name, 'the display name is not the identity').not.toMatch(HEX_UID);

			const source = page.getByTestId('console-entry').first().locator('button.node');
			await expect(source, 'the row attributes the raise to a node').toBeVisible();
			await expect(source).toHaveText(name);
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

test.describe('the node body seam', () => {
	/**
	 * The loudest line on a node card was internal to it.
	 *
	 * `GoofiNode`'s `.header` paints a `border-bottom` and `SlotViewer`'s `header` paints a
	 * `border-top`, with no margin between them — so the seam between a node's title bar and its first
	 * slot rendered at 2 CSS px, against the card's own 1 px outline. On the app's most-repeated
	 * element, that is the single biggest contributor to "too salient", and it is not a design choice
	 * anyone made: it is two components each drawing their own edge.
	 *
	 * The tree already contained the intended shape — `PlacementPreview`, documented as mirroring
	 * GoofiNode exactly, de-duplicates it with `.viewers .slot-row:first-child { border-top: none }`.
	 * This applies that precedent from the child's side, and pins the result in composited pixels
	 * rather than in the source, because "how many lines does this seam paint" has no DOM answer.
	 *
	 * The node header's own `border-bottom` deliberately SURVIVES, and is asserted here: `inputUnits`
	 * floors every node body at one unit, so a node with no outputs has no slot row to carry the line
	 * and would otherwise lose its header/body separation entirely.
	 */
	test('the node card draws one line under its header, not two', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		try {
			await waitForNode(page, uid);

			// The card carries no uid of its own; reach it through the slot that does.
			const card = page
				.locator('.goofi-node')
				.filter({ has: page.locator(`.slot-viewer[data-node="${uid}"]`) })
				.first();
			const slotHeader = page.locator(`.slot-viewer[data-node="${uid}"] header`).first();
			await expect(slotHeader, 'the node renders its output slot').toBeVisible();

			expect(
				await slotHeader.evaluate((el) => getComputedStyle(el).borderTopWidth),
				'the first slot row adds no second line under the node header'
			).toBe('0px');

			const nodeHeader = card.locator('.header').first();
			expect(
				await nodeHeader.evaluate((el) => getComputedStyle(el).borderBottomWidth),
				'the separation is still drawn — once — so an output-less node keeps it'
			).toBe('1px');

			// One health indicator, and it is the library's dot rather than a shape of the card's own —
			// the header used to swap a bespoke circle for a bespoke spinner depending on the stage.
			// It settles green: amber (still coming up) and red (errored / dead) are asserted where
			// they are deterministic, in the unit tests over `nodeHealth` and in the UI gallery.
			const dot = nodeHeader.locator('.ui-status-dot');
			await expect(dot, 'the node header carries exactly one status dot').toHaveCount(1);
			await expect(dot, 'a running node reads as running').toHaveClass(/t-ok/);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});
});

test.describe('the node rate readout', () => {
	test('selection brings the node update rate forward exactly like hover', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [260, 220]);
		try {
			await waitForNode(page, uid);
			const card = page.locator(`.svelte-flow__node[data-id="${uid}"]`);
			const node = card.locator('.goofi-node');
			const rate = node.locator('.rate');
			await expect(rate, 'the first node-stats push supplies the rate').toBeVisible({
				timeout: 15_000
			});

			const opacity = (): Promise<number> =>
				rate.evaluate((el) => parseFloat(getComputedStyle(el).opacity));

			await page.locator('.topbar').hover();
			await expect.poll(opacity, { message: 'unselected and unhovered, the rate rests quietly' }).toBe(0.3);

			await card.hover();
			await expect.poll(opacity, { message: 'hover brings the rate forward' }).toBe(0.85);
			const hovered = await opacity();

			await card.locator('.header').click();
			await expect(node, 'the node accepted the selection').toHaveClass(/selected/);
			await page.locator('.topbar').hover();
			await expect.poll(opacity, { message: 'selection keeps the rate forward after hover leaves' }).toBe(0.85);
			expect(await opacity(), 'selection and hover use the exact same emphasis').toBe(hovered);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});
});

test.describe('the canvas controls', () => {
	/**
	 * The node editor's on-canvas control cluster (SvelteFlow's `<Controls/>`).
	 *
	 * Two invariants, both of which the library's defaults get wrong for this app:
	 *
	 * 1. It offers viewport controls and nothing else. `<Controls/>` ships a fourth button — Toggle
	 *    Interactivity — which flips `nodesDraggable`/`nodesConnectable`/`elementsSelectable` off in
	 *    one click. goofi has no read-only mode, no other surface says the canvas is locked, and the
	 *    button is a lock glyph next to three magnifiers, so hitting it reads as "the editor broke".
	 * 2. It sits in the panel's corner, at a gap the rule actually DECLARES. Flow's own
	 *    `.svelte-flow__panel` adds `margin: 15px` underneath the offsets, so a rule reading
	 *    `bottom: 20px; left: 20px` rendered at 35px — a third of the way into a 412px phone canvas.
	 *
	 * The gap is asserted against the live root font-size, not a pixel literal: `--space-*` is rem and
	 * the `html` base is a responsive clamp, so the number moves with the viewport by design.
	 */

	test('the editor controls are the viewport controls, and nothing else', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const labels = await page
			.locator('.svelte-flow__controls button')
			.evaluateAll((els) => els.map((el) => el.getAttribute('aria-label')));
		expect(labels, 'zoom in, zoom out, fit — no interactivity lock').toEqual([
			'Zoom In',
			'Zoom Out',
			'Fit View'
		]);
	});

	test('the editor controls sit --space-8 into the panel corner on a fine pointer', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const { left, bottom, rem } = await controlsInset(page);
		// --space-8 = 1.5rem: clear of the 16px corner grips (which is what the inset is FOR), and the
		// same ~20px the old rule always claimed to draw before Flow's margin was subtracted out of it.
		expect(left, 'the gap is the declared one, with no library margin hiding inside it').toBeCloseTo(
			1.5 * rem,
			0
		);
		expect(bottom).toBeCloseTo(1.5 * rem, 0);
		expect(left, 'and it still clears the 16px corner grip').toBeGreaterThan(16);
	});
});

test.describe('where a context menu opens', () => {
	// Task 7's regression guard. `.panel-body` gains `container-type: inline-size` (the `@container`
	// enablement). The add-node menu (`.menu-anchor`) is `position: fixed`, positioned in VIEWPORT
	// coordinates (menuPos = window.innerWidth / clientX math), so it is portalled to <body>. Two
	// independent checks, because either alone has a blind spot:
	//   1. STRUCTURAL — the anchor really escaped the panel to <body>. This is the check with teeth
	//      against the concrete regression (someone drops `use:portal`): `inline-size` containment does
	//      NOT by itself trap fixed descendants in Chromium, so a position-only check would pass even
	//      un-portalled today — but it WOULD break the moment the body gained a real containing-block
	//      trigger (a transform/filter). Asserting the portal keeps the invariant that makes it safe.
	//   2. POSITIONAL — the menu lands at its intended viewport point (not shifted by the panel offset),
	//      which catches any real re-anchoring plus general placement bugs in openAddMenu.
	//
	// The panel-addressed path (`window.goofi`'s `openAddMenu`, which names a panel rather than a
	// point) centres the menu over that editor: `openAddMenu(r.left + r.width/2, r.top + 60, 'center')`,
	// where r is the editor root's viewport rect and the half-width comes from the MEASURED menu — so
	// the expected x below is the centre minus half of the menu's own 320px box.
	test('the add-node menu portals to <body> and opens at its intended viewport point', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);

		// The editor root (`.canvas-wrap` is the rootEl openAddMenuCentered measures).
		const editor = page.locator('.canvas-wrap').first();
		await editor.waitFor();
		const r = (await editor.boundingBox())!;
		const expected = { x: r.x + r.width / 2 - 160, y: r.y + 60 };

		// Open the add-node menu through the panel-addressed façade command.
		await page.evaluate(() => (window as any).goofi.commands.openAddMenu());

		const anchor = page.getByTestId('add-node-menu-anchor');
		await expect(anchor, 'the menu is shown').toBeVisible();

		// 1. STRUCTURAL: the anchor escaped `.panel-body` — its parent is <body>, and no `.panel-body`
		//    ancestor remains. Dropping `use:portal` fails this immediately.
		const escaped = await anchor.evaluate((el) => ({
			parentIsBody: el.parentElement === document.body,
			insidePanelBody: !!el.closest('.panel-body')
		}));
		expect(escaped.parentIsBody, 'the menu anchor is portalled directly to <body>').toBe(true);
		expect(escaped.insidePanelBody, 'the menu anchor is not left inside a panel body').toBe(false);

		// 2. POSITIONAL: the rendered box sits at the viewport-relative target (a few px for borders).
		const box = (await anchor.boundingBox())!;
		expect(Math.abs(box.x - expected.x), 'menu left is at the viewport target').toBeLessThanOrEqual(2);
		expect(Math.abs(box.y - expected.y), 'menu top is at the viewport target').toBeLessThanOrEqual(2);
	});

	// D-M2's other half: the add-node menu must SHARE the `clampToViewport` SSOT, not re-derive it (or,
	// on two of its paths, skip it). `Tab` in the active editor opens the 320px menu at the last
	// window-level pointer position — so with the cursor near a far corner the menu rendered off-screen
	// and the node could not be picked at all. RED before the four paths were routed through one
	// measured, clamped `openAddMenu`.
	test('Tab opens the add-node menu on-screen even with the pointer in the far corner', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const vp = page.viewportSize()!;

		// Make the editor the active panel (Tab is gated on it), then park the pointer in the corner —
		// `trackMouse` is a window listener, so this is the position `openMenuAtCursor` will use.
		await page.locator('.canvas-wrap').first().click({ position: { x: 20, y: 20 } });
		await page.mouse.move(vp.width - 3, vp.height - 3);
		await page.keyboard.press('Tab');

		const anchor = page.getByTestId('add-node-menu-anchor');
		await expect(anchor, 'Tab opened the menu').toBeVisible();
		const box = (await anchor.boundingBox())!;
		expect(box.x, 'the menu does not run off the left edge').toBeGreaterThanOrEqual(0);
		expect(box.y, 'the menu does not run off the top edge').toBeGreaterThanOrEqual(0);
		expect(box.x + box.width, 'the menu does not run off the right edge').toBeLessThanOrEqual(vp.width);
		expect(box.y + box.height, 'the menu does not run off the bottom edge').toBeLessThanOrEqual(
			vp.height
		);
		await page.keyboard.press('Escape');
		await expect(anchor).toBeHidden();
	});

	// The fourth entry point, and the only one whose geometry the unification restated: a TARGET port
	// hangs the menu to its LEFT (the click point is the menu's RIGHT edge + 12px of clearance). That
	// used to be a hardcoded `-332`; it is now the measured width, so this pins the resulting box.
	test('clicking an input port opens the add-node menu to the LEFT of the port', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Buffer', 'signal', [700, 200]);
		await waitForNode(page, uid);

		// The backend graph is shared across this worker's specs, so the node comes back out even if an
		// assertion below throws — a leaked node breaks the fs-browser round trip further down the file.
		try {
			const port = page
				.locator(`.svelte-flow__node[data-id="${uid}"]`)
				.getByTestId('slot-input')
				.first();
			await expect(port, 'the node rendered its input port').toBeVisible();
			const portBox = (await port.boundingBox())!;
			await port.click();

			const anchor = page.getByTestId('add-node-menu-anchor');
			await expect(anchor, 'the port click opened the seeded menu').toBeVisible();
			const box = (await anchor.boundingBox())!;
			const clickX = portBox.x + portBox.width / 2;
			expect(box.x + box.width, 'the menu ends 12px left of the port').toBeCloseTo(clickX - 12, 0);
			expect(box.x, 'and still starts on-screen').toBeGreaterThanOrEqual(6);

			await page.keyboard.press('Escape');
			await expect(anchor).toBeHidden();
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid);
		}
	});
});

test.describe('a context menu submenu', () => {
	/**
	 * The desktop reference for `workspace/ContextMenu.svelte`'s submenus, pinned because R-Task 7
	 * moved the hover listener from `mouseenter` to `pointerenter` (so the pointer type is knowable)
	 * and gave a parent row a click action it did not have.
	 *
	 * Both halves matter here: hovering must still expand — that is the fine-pointer behaviour and it
	 * is the reference — and clicking a parent must OPEN rather than toggle, or the click that follows
	 * a hover would close the very submenu the hover just opened.
	 */

	function openHeaderMenu(page: Page) {
		return page.getByTestId('panel-header').first().click({ button: 'right' });
	}

	const parentRow = (page: Page) =>
		page.locator('.context-menu .item').filter({ hasText: 'Change content' }).first();

	test('hovering a parent row still expands its submenu', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await openHeaderMenu(page);
		await expect(page.locator('.context-menu')).toHaveCount(1);

		await parentRow(page).hover();
		await expect(page.locator('.context-menu'), 'the submenu opened on hover').toHaveCount(2);
		await page.keyboard.press('Escape');
		await expect(page.locator('.context-menu')).toHaveCount(0);
	});

	test('clicking a parent row opens its submenu and never closes it', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await openHeaderMenu(page);

		// The hover that precedes any real click has already opened it; the click must leave it open.
		await parentRow(page).click();
		await expect(page.locator('.context-menu')).toHaveCount(2);
		await parentRow(page).click();
		await expect(page.locator('.context-menu'), 'a second click is not a toggle').toHaveCount(2);
		await page.keyboard.press('Escape');
		await expect(page.locator('.context-menu')).toHaveCount(0);
	});
});

test.describe('cancelling a gesture', () => {
	/**
	 * R Task 5 — a CANCELLED drag must leave nothing behind (§3.1g), and a panel has a PIXEL floor
	 * (D-R10). Both are desktop bugs, which is why they are pinned in the fine-pointer project: the
	 * corner grip is a fine-pointer gesture by R-Task 3's decision, and a narrow desktop window
	 * collapses a panel exactly the way a phone does.
	 *
	 * The corner grip armed `pointermove`/`pointerup` on the window and detached on `pointerup` ALONE.
	 * A pointer can be CANCELLED instead of released — the browser reclaims the gesture for a pan, a
	 * system UI takes over, the owning panel unmounts mid-drag — and then the listeners survived, the
	 * preview ghost stayed painted, and the NEXT release anywhere in the app committed the abandoned
	 * intent: a `ws.split()` or a `ws.close()` restructuring the workspace one click later (H6).
	 *
	 * The cancel is dispatched rather than provoked, because provoking it needs a browser decision this
	 * harness cannot make. `ui/dragGesture.ts` owns the state machine (unit-tested); this proves the
	 * real grip is wired to it.
	 */

	const panels = (page: Page) => page.locator('[data-panel-id]');

	test('a cancelled corner drag drops its ghost and never commits a split', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const before = await panels(page).count();

		// The grip is a 16px box clipped to the panel body's top-left triangle.
		const body = (await page.locator('.panel-body').first().boundingBox())!;
		const grip = { x: Math.round(body.x + 3), y: Math.round(body.y + 3) };

		try {
			await page.mouse.move(grip.x, grip.y);
			await page.mouse.down();
			// Past THRESHOLD (24px), which is what arms a split intent and paints the preview.
			await page.mouse.move(grip.x + 160, grip.y + 20);
			await expect(page.locator('.drag-ghost'), 'the drag is really in flight').toHaveCount(1);

			await page.locator('.panel-body .corner.tl').first().dispatchEvent('pointercancel');
			await expect(
				page.locator('.drag-ghost'),
				'a cancelled gesture stops previewing a result it will not produce'
			).toHaveCount(0);

			// The stale-commit half: the release used to run the leaked window `pointerup`.
			await page.mouse.up();
			await page.waitForTimeout(150);
			expect(await panels(page).count(), 'and it commits nothing when the button comes up').toBe(
				before
			);
		} finally {
			// If the leak fired, the split is a tracked layout command — undo it, so the shared workspace
			// (and the 400ms layout push that outlives this page) is handed back as it was found.
			for (let i = 0; i < 4 && (await panels(page).count()) > before; i++) {
				await page.evaluate(() => (window as any).goofi.commands.undo());
				await page.waitForTimeout(150);
			}
			await page.waitForTimeout(700);
		}
	});

	test('a panel cannot be dragged below its pixel floor', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const before = await panels(page).count();

		// Split through the header menu — the same route `panel-surface.spec.ts` uses.
		await page.getByTestId('panel-header').first().click({ button: 'right' });
		await page.locator('.context-menu .item', { hasText: 'Split Right' }).first().click();
		await expect(panels(page)).toHaveCount(before + 1);

		try {
			const seam = (await page.locator('.splitter.row').first().boundingBox())!;
			const at = { x: Math.round(seam.x + seam.width / 2), y: Math.round(seam.y + seam.height / 2) };

			// Drag the seam hard into the left wall. MIN_FRACTION alone is 5% of whatever the window
			// happens to be — 64px here, and a 20px sliver on a 412px phone. MIN_PANEL_PX is what stops
			// it at something that is still recognisably a panel.
			await page.mouse.move(at.x, at.y);
			await page.mouse.down();
			for (let x = at.x; x > 0; x -= 60) await page.mouse.move(x, at.y);
			await page.mouse.move(-500, at.y);
			await page.mouse.up();

			const w = await panels(page)
				.first()
				.evaluate((el) => el.getBoundingClientRect().width);
			expect(
				w,
				'the floor is a size, not a percentage of whatever the window happens to be'
			).toBeGreaterThanOrEqual(115);
		} finally {
			await page
				.getByTestId('panel-header')
				.nth(1)
				.getByRole('button', { name: 'Close panel' })
				.click();
			await expect(panels(page), 'the workspace is back to one panel').toHaveCount(before);
		}
	});
});

test.describe('what the keyboard reaches', () => {
	/**
	 * Who owns a key press — the editor panel, the browser, or the modal on top of it.
	 *
	 * `NodeEditorPanel`'s window-level `onKeydown` guards on two things only: is my panel the active
	 * one, and is the target a text field. Neither question is the one that matters at these two
	 * boundaries, and both were carried over from the M audit (R2-3 and R2-2) without ever entering a
	 * task brief:
	 *
	 *  - **Tab** was `preventDefault`ed for EVERY non-field target, with no `shiftKey` check. The first
	 *    Tab looks fine because it opens the add-node menu and the menu focuses its search — but
	 *    `{#if menuOpen}` is unkeyed, so re-entering `openAddMenu` with the menu already open neither
	 *    remounts it nor re-fires that focus. Every later Tab is a bare `preventDefault` with nothing
	 *    to refocus: a one-way trap (WCAG 2.1.2), and the reason no chrome outside the canvas — the tab
	 *    strip, the header actions, the inspector's ✕, R's own `.conn-label` focus reveal — was ever
	 *    Tab-reachable.
	 *  - The whole handler ignored the fact that a modal `<dialog>` was up. It opens with focus on its
	 *    path field, which the allowlist does cover — but the first navigation click puts focus on a
	 *    `<button>`, and load mode REQUIRES one (`fs-open` is `disabled={!selected}`). From there
	 *    Escape dismissed the dialog AND ran the editor's own Escape ladder behind it, and Ctrl+A
	 *    selected every node on the hidden canvas.
	 *
	 * The standdown is read from the DOM (`closest('dialog[open]')`), NOT from `ui().modalOpen`: that
	 * flag is a ref-count an in-panel fx editor raises too, and standing the canvas down for a merely
	 * expanded inspector field would be a different bug. The last test here is that trade, pinned.
	 */

	/** A short, stable name for whatever currently has focus. */
	const activeName = (page: Page): Promise<string | null> =>
		page.evaluate(() => {
			const el = document.activeElement as HTMLElement | null;
			if (!el) return null;
			return el.dataset.testid ?? el.tagName.toLowerCase();
		});

	const selectedNodes = (page: Page): Promise<string[]> =>
		page.evaluate(() => (window as any).goofi.query.selection().nodes);

	/**
	 * Open the Load browser and leave focus on a real control inside it.
	 *
	 * `showModal()` lands on the path field, which the handler's INPUT allowlist happens to cover — so
	 * the reachable state is one navigation click later. Clicking a root is the first thing load mode
	 * asks of anyone, and every route through it (a root, a directory row, a `.gfi` row) ends with
	 * focus on a `<button>`.
	 */
	async function openBrowserOnAButton(page: Page): Promise<ReturnType<Page['getByTestId']>> {
		await page.getByTestId('topbar-load').click();
		const modal = page.getByTestId('fs-browser');
		await expect(modal).toBeVisible();
		await modal.locator('nav.roots button.root').first().click();
		expect(await activeName(page), 'focus is on one of the dialog’s buttons, not its path field').toBe(
			'button'
		);
		return modal;
	}

	test('Tab on a chrome control is the browser’s, not the canvas’s', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// The node editor is the active panel from the first frame (DEFAULT_PANEL_TYPE + `_focusFirst`),
		// and no keyboard event ever clears that — which is why this reproduces in the boot layout.
		await page.getByTestId('topbar-load').focus();
		await page.keyboard.press('Tab');

		await expect(
			page.getByTestId('add-node-menu-anchor'),
			'the canvas does not claim a Tab it was not given'
		).toHaveCount(0);
		expect(await activeName(page), 'focus moved on to the next control in the header').toBe(
			'topbar-overflow'
		);
	});

	test('focus can leave the add-node menu again', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
		const menu = page.getByTestId('add-node-menu-anchor');
		await expect(menu).toBeVisible();
		await expect(page.getByTestId('add-menu-search'), 'the menu takes focus on open').toBeFocused();

		const outside = (): Promise<boolean> =>
			page.evaluate(() => {
				const a = document.activeElement;
				const m = document.querySelector('[data-testid="add-node-menu-anchor"]');
				return !!a && !!m && !m.contains(a);
			});
		let left = false;
		// Bounded: the menu holds a search field plus one row per palette entry, so a few dozen presses
		// is generous. Without the fix this never leaves — the second Tab lands on a row <button> and
		// every press after it is a bare preventDefault.
		for (let i = 0; i < 60 && !left; i++) {
			await page.keyboard.press('Tab');
			left = await outside();
		}
		expect(left, 'Tab is not a one-way door into the palette (WCAG 2.1.2)').toBe(true);

		await page.keyboard.press('Escape');
		await expect(menu).toHaveCount(0);
	});

	test('Tab on the bare canvas still opens the add-node menu', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		// The Blender-style shortcut is the point of the branch and must survive its scoping: with
		// nothing focused, the canvas is what the key press is for.
		await page.keyboard.press('Tab');
		const menu = page.getByTestId('add-node-menu-anchor');
		await expect(menu).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(menu).toHaveCount(0);
	});

	test('Escape dismisses the file browser without also running the editor’s ladder', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			await selectNode(page, uid);
			expect(await selectedNodes(page)).toEqual([uid]);

			const modal = await openBrowserOnAButton(page);

			await page.keyboard.press('Escape');
			await expect(modal, 'the reflex dismissal closes the dialog').toBeHidden();
			expect(
				await selectedNodes(page),
				'…and only the dialog — the canvas behind it kept its selection'
			).toEqual([uid]);
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});

	/**
	 * The chord the panel's own guard structurally cannot reach. Deletion is deliberately delegated OUT
	 * of `onKeydown` to SvelteFlow (`deleteKey={['Delete','Backspace']}` + `ondelete`), and its
	 * `KeyHandler` is a bare `<svelte:window>` keydown whose only filter is `isInputDOMNode` — neither
	 * "is my panel active" nor "is a dialog up". The browser's roots and entries are plain `<button>`s,
	 * so from the state `openBrowserOnAButton` leaves, Backspace — the reflex a file browser trains for
	 * "go up a folder" — ran `deleteElements` on the canvas behind an opaque modal. Ctrl+Z is itself
	 * stood down while the dialog is up, so it could not even be reversed in place.
	 *
	 * The standdown goes on the MODAL (`.nokey`, which xyflow's `isInputDOMNode` honours via
	 * `closest`), not as a second condition on the editor: FsBrowser is the app's only real modal, and
	 * the surface that takes the keyboard is the one that should say so.
	 */
	for (const key of ['Backspace', 'Delete'] as const) {
		test(`${key} does not delete the selection behind the file browser`, async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
			await waitForNode(page, uid);
			try {
				await selectNode(page, uid);
				expect(await selectedNodes(page)).toEqual([uid]);

				const modal = await openBrowserOnAButton(page);
				await page.keyboard.press(key);
				// A non-event needs a settle window; the delete is a command round-trip, so give it one
				// far longer than the trip takes.
				await page.waitForTimeout(500);
				expect(
					await page.evaluate(
						(u) => ((window as any).goofi.query.graph().nodes as Array<{ uid: string }>).some((n) => n.uid === u),
						uid
					),
					'the canvas behind the modal still holds the node'
				).toBe(true);

				await page.keyboard.press('Escape');
				await expect(modal).toBeHidden();
			} finally {
				await page.evaluate(() => (window as any).goofi.commands.clearSelection());
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
		});
	}

	test('Ctrl+A does not reach the canvas behind the file browser', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		try {
			await openBrowserOnAButton(page);
			await page.keyboard.press('Control+a');
			// A non-event needs a settle window; the selection write is synchronous, so this is orders of
			// magnitude more than it would need.
			await page.waitForTimeout(300);
			expect(await selectedNodes(page), 'the hidden canvas selected nothing').toEqual([]);
		} finally {
			await page.keyboard.press('Escape');
			await expect(page.getByTestId('fs-browser')).toBeHidden();
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});

	/**
	 * The trade the modal guard must NOT overshoot. `ui().modalOpen` is a ref-count that
	 * `inspector/ParamField.svelte` raises for a merely-expanded fx editor, so the naive
	 * `if (ui().modalOpen) return;` would stand the canvas down while an inspector field is open —
	 * which is why R2-2 was deferred to R in the first place. A dialog is a dialog; an fx editor is a
	 * panel that happens to hold a textarea.
	 */
	test('an expanded fx editor holds the global standdown but not the editor’s own keys', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const osc = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, osc);
		const buf = await addNode(page, 'Buffer', 'signal', [320, 40]);
		await waitForNode(page, buf);
		try {
			await selectNode(page, osc);
			const pane = page.getByTestId('auto-side-panel');
			await expect(pane).toHaveClass(/open/);
			await pane.getByTestId('param-fx-toggle').first().click();
			await pane.getByTestId('param-expr-expand').first().click();
			await expect(pane.getByTestId('param-expr-multiline')).toBeVisible();
			await expect
				.poll(() => page.evaluate(() => (window as any).goofi.query.modalOpen()))
				.toBe(true);

			// Focused, not clicked: a click on the collapse chip would close the very editor under test.
			await pane.getByTestId('param-expr-collapse').first().focus();
			await page.keyboard.press('Control+a');
			// `arrayContaining`, not `toEqual`: Select all takes the whole scope, and the suite's one
			// shared backend carries whatever earlier specs left on the canvas. What is under test is that
			// the chord RAN — before it, only `osc` was selected.
			await expect
				.poll(() => selectedNodes(page), {
					message: 'the editor’s own Select all still fires with an fx editor expanded'
				})
				.toEqual(expect.arrayContaining([osc, buf]));
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [osc, buf]);
			await waitForNoNode(page, osc).catch(() => {});
			await waitForNoNode(page, buf).catch(() => {});
		}
	});
});

test.describe('undo from the canvas', () => {
	/**
	 * Undo as a BUTTON reaches it. That undo/redo work at all — that a stack walks back to an empty
	 * patch and forward to the same uids — is the Rust suite's (`editing.rs`), driven through the one
	 * op vocabulary with no browser in the way. What can only be asked here is whether a control in
	 * the app records a step at all.
	 */


	/** The bound node of the sole panel, as the façade reports it. */
	function boundNode(page: import('@playwright/test').Page): Promise<string | null> {
		return page.evaluate(() => (window as any).goofi.query.panels()[0].node);
	}

	/**
	 * The unlink ✕ is an EDIT — it dirties the patch — but it used to write the panel state straight
	 * into the layout tree with no history entry of its own. Layout undo restores a whole
	 * `WorkspaceState` snapshot captured at the last tracked action, so an unrecorded write landing
	 * after one is in neither `before` nor `after`: an unrelated Ctrl+Z destroyed it and the redo did
	 * not bring it back. The store op is what the vitest pins; this pins the BUTTON reaching it.
	 */
	test('the unlink ✕ earns its own undo step, which restores the binding', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const panelId: string = await page.evaluate(
			() => (window as any).goofi.query.panels()[0].panelId
		);
		const uid = await addNode(page, 'Oscillator', 'inputs');
		await waitForNode(page, uid);
		try {
			await page.evaluate(
				([id, u]) => {
					(window as any).goofi.commands.setPanelType(id, 'parameters');
					(window as any).goofi.commands.bindNodeToPanel(id, u);
				},
				[panelId, uid] as const
			);
			// A panel write is a COMMAND now, so the binding appears when the manager's delta does —
			// polled, like every other assertion in this test.
			await expect.poll(() => boundNode(page), { message: 'the panel is bound' }).toBe(uid);

			await page.getByTestId('node-linked-panel').getByRole('button', { name: 'Unlink node' }).click();
			await expect.poll(() => boundNode(page), { message: 'the ✕ unbinds' }).toBe(null);
			expect(
				await page.evaluate(() => (window as any).goofi.query.undoLabel()),
				'the click named its own undo step'
			).toBe('Unbind node from panel');

			await undo(page);
			await expect.poll(() => boundNode(page), { message: 'undo re-binds' }).toBe(uid);
			await redo(page);
			await expect.poll(() => boundNode(page), { message: 'redo unbinds again' }).toBe(null);
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

test.describe('entering and leaving a sub-patch', () => {
	/**
	 * Entering a sub-patch by double-clicking its group node — the DESKTOP half (the coarse door is
	 * `touch-editor.spec.ts`).
	 *
	 * `enterInstance` has exactly one caller: a hand-rolled double-click recogniser in
	 * `NodeEditorPanel.svelte`, because neither SvelteFlow's `onnodeclick` (it suppresses the second
	 * click) nor the native `dblclick` (the first click rebuilds `flowNodes` and detaches the element)
	 * survives the first click. What that recogniser did NOT do was re-resolve the second click's
	 * target: it matched on COORDINATES alone. So the first click selects the instance, the inspector
	 * slides in over 120ms and covers all but `--hit` of the editor — including the very node the
	 * second click has to hit — and the second click both entered the instance AND actuated whatever
	 * inspector control had arrived under the pointer. On a sub-patch that control is
	 * `subpatch-expand-inspector`, which DISSOLVES it.
	 *
	 * Not a touch defect: this reproduces at a normal ~150ms desktop double-click. The suite hid it
	 * only because Playwright's `.dblclick()` fires both clicks ~0ms apart, before the pane has begun
	 * to slide.
	 */

	/** The editor viewport's flow→screen scale (its CSS transform's `a` component). */
	function flowScale(page: Page): Promise<number> {
		return page
			.locator('.svelte-flow__viewport')
			.first()
			.evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).a);
	}

	const nodeBox = (page: Page, uid: string) =>
		page.locator(`.svelte-flow__node[data-id="${uid}"]`).boundingBox();

	/** The flow-node id actually on top at a viewport point, or null. */
	const nodeAt = (page: Page, at: { x: number; y: number }): Promise<string | null> =>
		page.evaluate(
			(p) =>
				(document.elementFromPoint(p.x, p.y) as HTMLElement | null)
					?.closest('.svelte-flow__node')
					?.getAttribute('data-id') ?? null,
			at
		);

	/** The inspector pane's current horizontal slide offset in px (0 = fully in). */
	const paneOffset = (page: Page): Promise<number> =>
		page
			.getByTestId('auto-side-panel')
			.evaluate((el) => Math.round(new DOMMatrixReadOnly(getComputedStyle(el).transform).e));

	/** Reposition `uid` (currently at `from` in flow space) so its rendered CENTRE lands on `target`. */
	async function centreNodeOn(
		page: Page,
		uid: string,
		from: [number, number],
		target: { x: number; y: number }
	): Promise<void> {
		const box = (await nodeBox(page, uid))!;
		const s = await flowScale(page);
		const next: [number, number] = [
			Math.round(from[0] + (target.x - (box.x + box.width / 2)) / s),
			Math.round(from[1] + (target.y - (box.y + box.height / 2)) / s)
		];
		await page.evaluate(
			([u, p]) => (window as any).goofi.commands.setNodePos(u, p),
			[uid, next] as const
		);
		await expect
			.poll(async () => {
				const b = await nodeBox(page, uid);
				return b ? Math.round(Math.abs(b.x + b.width / 2 - target.x)) : 999;
			}, { message: 'the group node moved under the inspector' })
			.toBeLessThanOrEqual(2);
	}

	/** A sub-patch with one member, plus a link across the cut so it owns a real interface. */
	async function makeSubPatch(page: Page): Promise<{ osc: string; buf: string; inst: string }> {
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
		const inst: string = await page.evaluate(
			(b) => (window as any).goofi.commands.groupNodes([b], [200, 200]),
			buf
		);
		await expect(page.getByTestId('subpatch-node')).toBeVisible();
		return { osc, buf, inst };
	}

	const instances = (page: Page): Promise<Record<string, unknown>> =>
		page.evaluate(() => (window as any).goofi.query.instances());

	/** The direct member uids of a sub-patch, as the editor's forest reports them. */
	async function scopeMembers(page: Page, inst: string): Promise<string[]> {
		const rec = (await instances(page))[inst] as { members?: Record<string, unknown> } | undefined;
		return Object.keys(rec?.members ?? {});
	}

	/** Leave the sub-patch this editor is inside, if it is inside one. */
	async function exitSubPatch(page: Page): Promise<void> {
		const crumb = page.getByTestId('subpatch-breadcrumb');
		if (await crumb.isVisible()) await crumb.getByRole('button', { name: 'Patch', exact: true }).click();
	}

	test('a double-click enters a sub-patch parked under the inspector without dissolving it', async ({
		page
	}) => {
		// The pane's 120ms slide is a hazard in BOTH directions here — it is still covering the node
		// while it parks, and still off it while it arrives — and neither is what this test is about.
		// The app's own reduced-motion rule collapses every transition to 0.01ms, so the pane's position
		// is a function of the selection alone and the click interval below is free to be a realistic
		// double-click rather than a race against a transition.
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const { osc, buf, inst } = await makeSubPatch(page);
		try {
			// Learn the inspector's footprint from the pane it opens for THIS node, then park the group
			// node exactly under its one structural action — the button a stray second click actuates.
			await selectNode(page, inst);
			const expand = page.getByTestId('subpatch-expand-inspector');
			await expect(expand).toBeVisible();
			// Measured only once the 120ms slide has SETTLED — mid-transition the pane's translateX still
			// puts the button hundreds of px off the right edge of the window.
			await expect.poll(() => paneOffset(page)).toBe(0);
			const eb = (await expand.boundingBox())!;
			const target = { x: Math.round(eb.x + eb.width / 2), y: Math.round(eb.y + eb.height / 2) };
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await expect(page.getByTestId('auto-side-panel')).not.toHaveClass(/open/);
			await centreNodeOn(page, inst, [200, 200], target);
			// Stated as a precondition, not assumed: the FIRST click has to reach the node, so the node
			// has to be what is actually on top at that point.
			await expect
				.poll(() => nodeAt(page, target), { message: 'the group node is the topmost thing there' })
				.toBe(inst);

			// A normal double-click: two clicks at one point, ~120ms apart.
			await page.mouse.click(target.x, target.y);
			await page.waitForTimeout(120);
			await page.mouse.click(target.x, target.y);

			// Settled, not sampled. Today the gesture enters the instance AND actuates Expand under the
			// pointer, and the dissolve is an RPC round-trip — so the breadcrumb really does appear for a
			// frame or two before the climb-out effect pops it. Asserting the END state is what tells the
			// two outcomes apart.
			await page.waitForTimeout(500);
			expect(
				Object.keys(await instances(page)),
				'the sub-patch survived the gesture (the inspector control under the pointer did not fire)'
			).toContain(inst);
			await expect(
				page.getByTestId('subpatch-breadcrumb'),
				'and the double-click entered it'
			).toBeVisible();
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await exitSubPatch(page);
			if (Object.keys(await instances(page)).includes(inst))
				await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
			await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), [osc, buf]);
			await waitForNoNode(page, osc).catch(() => {});
			await waitForNoNode(page, buf).catch(() => {});
		}
	});

	/**
	 * Wait until the editor camera has stopped moving.
	 *
	 * `enterInstance` schedules a `fitView` 60ms after it descends, and `fitView` is implemented by
	 * CLICKING SvelteFlow's own Controls button — a synthetic click inside `.canvas-wrap`, which is
	 * exactly what `PlacementPreview` listens for to commit. Opening the add menu before that fit has
	 * landed lets it commit the ghost early, at the wrong point, and the test's own placement click then
	 * falls through to the pane (clearing the selection). Settled = the viewport transform holds still,
	 * sampled only after the deferred fit is due.
	 */
	async function cameraSettled(page: Page): Promise<void> {
		await page.waitForTimeout(150); // outlast the deferred fit; the poll below covers its animation
		let prev = '';
		await expect
			.poll(
				async () => {
					const now = await page
						.locator('.svelte-flow__viewport')
						.first()
						.evaluate((el) => getComputedStyle(el).transform);
					const held = now === prev;
					prev = now;
					return held;
				},
				{ message: 'the editor camera stopped moving' }
			)
			.toBe(true);
	}

	/**
	 * The other half of entering: what you ADD once you are inside.
	 *
	 * `add_node` carried the entered scope from the panel all the way to the bridge and the bridge threw
	 * it away, so the node was minted at ROOT. The canvas renders only `childrenOfScope(entered)`, so it
	 * was placed into a scope that does not draw it — while the panel still auto-selected it, leaving the
	 * inspector editing a node nowhere on screen. Driven through the real door (the add menu + a click to
	 * place), because the auto-selection is the panel's, not the façade's.
	 */
	test('a node added inside an entered sub-patch lands in that scope, on screen and selected', async ({
		page
	}) => {
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await waitForApp(page);
		const { osc, buf, inst } = await makeSubPatch(page);
		let added: string | undefined;
		try {
			// Playwright's `.dblclick()` fires both clicks ~0ms apart — before the inspector can slide over
			// the node — so entering is not itself the race the spec above isolates.
			await page.getByTestId('subpatch-node').dblclick();
			await expect(page.getByTestId('subpatch-breadcrumb'), 'the sub-patch is entered').toBeVisible();
			await cameraSettled(page);

			// Add through the REAL door — the add-node menu, then a click to place the ghost — because the
			// auto-selection under test is the PANEL's, not the façade's. The type is picked from the
			// search field rather than by clicking its row: the unfiltered menu is a scrolling, grouped
			// list, so a row click is a hit-test against a box that can still be moving, and it
			// intermittently landed off the row (menu dismissed, nothing picked). Enter takes the
			// top-ranked match, asserted below, and needs no geometry at all.
			const before = new Set((await nodes(page)).map((n) => n.uid));
			await page.evaluate(() => (window as any).goofi.commands.openAddMenu());
			const search = page.getByTestId('add-menu-search');
			await expect(search, 'the menu is open and holds the keyboard').toBeFocused();
			await search.fill('Buffer');
			await expect(
				paletteItem(page, 'Buffer'),
				'the exact name match leads the ranked list, so Enter picks it'
			).toHaveClass(/\bhl\b/);
			await search.press('Enter');

			const ghost = page.getByTestId('placement-ghost');
			await expect(ghost, 'the Buffer ghost is pending').toBeVisible();
			const spot = await emptySpot(page);
			await page.mouse.click(spot.x, spot.y);
			await expect(ghost, 'the click placed it').toHaveCount(0);

			await expect
				.poll(async () => (await nodes(page)).filter((n) => !before.has(n.uid)).length, {
					message: 'one new node landed'
				})
				.toBe(1);
			const fresh = (await nodes(page)).find((n) => !before.has(n.uid))!;
			expect(fresh.type, 'and it is the type that was picked').toBe('Buffer');
			added = fresh.uid;

			// The defect as the canvas states it: this editor draws only the entered scope's children, so
			// a node that went to ROOT is simply not here.
			await expect(
				page.locator(`.svelte-flow__node[data-id="${added}"]`),
				'the new node renders inside the entered sub-patch'
			).toBeVisible();
			// And the forest agrees it is a DIRECT member — not a root node this canvas happens to draw.
			expect(await scopeMembers(page, inst), 'a direct member of the entered scope').toContain(added);
			// The panel auto-selects what it just placed, so the inspector is showing THIS node. Polled,
			// not sampled: the panel selects only once its `add_node` RPC has resolved, and the doc echo
			// that renders the node can land first.
			await expect
				.poll(
					async () => (await page.evaluate(() => (window as any).goofi.query.selection())).nodes,
					{ message: 'the placed node is the one selected' }
				)
				.toEqual([added]);
		} finally {
			await page.evaluate(() => (window as any).goofi.commands.clearSelection());
			await exitSubPatch(page);
			if (Object.keys(await instances(page)).includes(inst))
				await page.evaluate((i) => (window as any).goofi.commands.expandInstance(i), inst);
			const leftovers = [osc, buf, ...(added ? [added] : [])];
			await page.evaluate((ids) => (window as any).goofi.commands.removeNodes(ids), leftovers);
			for (const uid of leftovers) await waitForNoNode(page, uid).catch(() => {});
		}
	});
});

test.describe('the whole authoring pass, on a desktop', () => {
	/**
	 * The phone has one of these (`touch.spec.ts`) and the desktop did not, which matters because
	 * the two journeys differ at exactly the step this covers. Placing a node SELECTS it, so neither
	 * platform has to open the inspector by hand on the way in — and that is how every inspector
	 * spec came to reach its state through `commands.select`. On a desktop the user clicks the card,
	 * and until now nothing drove that click. End-to-end with a skip in the middle of it.
	 *
	 * Every door here is one a user has: Tab opens the palette, Enter picks the ranked type, a click
	 * commits the ghost, a click on bare canvas deselects, a click on the card selects again, and
	 * the parameter is typed into the rendered control.
	 */
	test('add a node from the palette, click it, and edit a parameter through its inspector', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		const inspector = page.getByTestId('auto-side-panel');

		// --- 1. ADD, through the palette ---------------------------------------------------------
		// The type is picked from the search field rather than by clicking its row: the unfiltered
		// menu is a scrolling, grouped list, so a row click hit-tests a box that can still be moving.
		await page.keyboard.press('Tab');
		const search = page.getByTestId('add-menu-search');
		await expect(search, 'Tab opened the palette and it holds the keyboard').toBeFocused();
		await search.fill('Oscillator');
		await expect(paletteItem(page, 'Oscillator'), 'the exact match leads the ranked list').toHaveClass(
			/\bhl\b/
		);
		await search.press('Enter');
		const ghost = page.getByTestId('placement-ghost');
		await expect(ghost, 'the Oscillator ghost is pending').toBeVisible();
		const spot = await emptySpot(page);
		await page.mouse.click(spot.x, spot.y);
		await expect(ghost, 'the click placed it').toHaveCount(0);
		await expect.poll(async () => (await nodes(page)).length, { message: 'a node landed' }).toBe(1);
		const osc = (await nodes(page))[0];

		try {
			// --- 2. DESELECT, so the click below is what opens the pane --------------------------
			// Placing already selected it. Click bare canvas to put that back, or step 3 would assert
			// a pane that was open before it ever pressed anything.
			await expect(inspector, 'placing a node inspects it').toHaveClass(/open/);
			await page.locator('.svelte-flow__pane').click({ position: { x: spot.x + 320, y: spot.y } });
			await expect(inspector, 'clicking off the node closes the inspector').not.toHaveClass(/open/);

			// --- 3. SELECT, by clicking the card -------------------------------------------------
			await selectNode(page, osc.uid);
			await expect(inspector, 'the click opened the inspector').toHaveClass(/open/);
			await expect(inspector.getByTestId('node-name')).toHaveText(osc.name);

			// --- 4. EDIT, through the rendered control -------------------------------------------
			const amp = inspector.getByTestId('param-field-amplitude').getByTestId('param-number');
			await amp.fill('0.37');
			await amp.press('Enter');
			await expect
				.poll(async () => (await nodeParams(page, osc.uid))?.oscillator?.amplitude?.value, {
					message: 'the typed value reached the backend and came back through the doc'
				})
				.toBeCloseTo(0.37, 5);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), osc.uid);
			await waitForNoNode(page, osc.uid).catch(() => {});
		}
	});
});
