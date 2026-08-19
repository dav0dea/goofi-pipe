// Watching a node's output: the viewer's own settings, its readouts, and the GL path.

import { test, expect, type Page } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { addNode, waitForNode, waitForNoNode } from '../lib/goofi';
import { VIEWER_HOVER_SURFACES, hoverSettled, surfaceStyles, unhover } from '../lib/viewerChrome';

test.describe('viewer settings', () => {
	/**
	 * The per-slot viewer settings cog — the second product consumer of the `Popover` primitive, and
	 * the half of spec §6's "add e2e where M changes real behaviour" that was never written (the
	 * ErrorPanel half was).
	 *
	 * Two things are pinned here. First the surface itself: the migrated menu must still render one
	 * `Field` per visible setting and round-trip a change through the ViewBinding. Second — the reason
	 * this file exists — its DISMISSAL. Pre-migration the menu portalled a full-screen backdrop that
	 * SWALLOWED the dismissing click; the Popover migration dropped it, leaving only a bubble-phase
	 * window listener, so a dismiss-click also acted on whatever was under it: it collapsed the very
	 * viewer being configured (persisted, with no undo entry), or — because a slot header stops
	 * pointerdown from ever reaching `window` to keep SvelteFlow from starting a node drag — toggled the
	 * other slot while leaving the menu open at a now-stale anchor.
	 *
	 * Nodes are torn down after each test so the shared backend graph stays clean for later specs.
	 */
	test.describe('viewer settings menu (Popover + catcher)', () => {
		let created: string[] = [];

		test.afterEach(async ({ page }) => {
			for (const uid of created) {
				await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
				await waitForNoNode(page, uid).catch(() => {});
			}
			created = [];
		});

		/** An Oscillator on the canvas; its single ARRAY output slot renders expanded by default. */
		async function oscillator(page: Page, pos: [number, number]): Promise<string> {
			const uid = await addNode(page, 'Oscillator', 'inputs', pos);
			created.push(uid);
			await waitForNode(page, uid);
			return uid;
		}

		function slot(page: Page, uid: string) {
			return page.locator(`.slot-viewer[data-node="${uid}"]`);
		}

		/** Click at a target's screen position with the RAW mouse, not `locator.click()`. Playwright's
		 * actionability check refuses to click an element another layer covers — which is the entire
		 * point here: the user aims at the header, and the catcher is what must receive the click. */
		async function clickAt(page: Page, target: ReturnType<typeof slot>): Promise<void> {
			const box = (await target.boundingBox())!;
			await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
		}

		test('the cog opens a Field per setting, and a change round-trips through the binding', async ({
			page
		}) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await oscillator(page, [40, 40]);
			const menu = page.getByTestId('viewer-settings-menu');

			await expect(menu, 'closed by default').toBeHidden();
			await slot(page, uid).getByTestId('viewer-settings-cog').click();
			await expect(menu, 'the cog opens the settings popover').toBeVisible();

			// The `line` schema's visible settings: Log X, Log Y, Auto, Show points (the manual Y min/max
			// are `showWhen`-gated behind Auto). Each is a Field, not a bespoke row.
			await expect(menu.locator('.ui-field'), 'one Field per visible setting').toHaveCount(4);
			const points = menu.locator('.ui-field').filter({ hasText: 'Show points' });
			const toggle = points.locator('input[type=checkbox]');
			await expect(toggle).not.toBeChecked();

			await toggle.click();
			await page.keyboard.press('Escape');
			await expect(menu).toBeHidden();

			// Re-opening re-reads the binding (the popover's content is unmounted while closed), so the
			// setting surviving the round trip is the binding having taken the write.
			await slot(page, uid).getByTestId('viewer-settings-cog').click();
			await expect(
				menu.locator('.ui-field').filter({ hasText: 'Show points' }).locator('input[type=checkbox]'),
				'the setting round-tripped through the ViewBinding'
			).toBeChecked();
			await page.keyboard.press('Escape');
			await expect(menu).toBeHidden();
		});

		// A wrong role is worse than none for a screen reader. `role="menu"` promises menuitems, and this
		// surface's content is a ScrollArea → Disclosure (a `<button aria-expanded aria-controls>`) →
		// Field + Toggle/Select/NumberInput: a settings FORM, with not one menuitem in it and no
		// accessible name. `Popover` imposes no role of its own precisely so the consumer declares the
		// fitting one. And the open state — plain to a sighted user from the surface being there — is
		// exposed on the trigger nowhere in this app today.
		test('the surface declares what it is, and the cog reports whether it is open', async ({
			page
		}) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await oscillator(page, [40, 40]);
			const menu = page.getByTestId('viewer-settings-menu');
			const cog = slot(page, uid).getByTestId('viewer-settings-cog');

			await expect(cog, 'the trigger reports its collapsed state').toHaveAttribute(
				'aria-expanded',
				'false'
			);
			await cog.click();
			await expect(cog, 'and its expanded one').toHaveAttribute('aria-expanded', 'true');
			await expect(menu, 'a NAMED group of settings, which is what it is').toHaveAttribute(
				'role',
				'group'
			);
			await expect(menu).toHaveAttribute('aria-label', 'viewer settings');
			await expect(menu.getByRole('menuitem'), 'it never owned a menuitem to be a menu of').toHaveCount(
				0
			);

			await page.keyboard.press('Escape');
			await expect(menu).toBeHidden();
			await expect(cog, 'and back to collapsed').toHaveAttribute('aria-expanded', 'false');
		});

		test('a pointerdown on another slot dismisses the menu without toggling that slot', async ({
			page
		}) => {
			await page.goto('/');
			await waitForApp(page);
			const a = await oscillator(page, [40, 40]);
			const b = await oscillator(page, [40, 260]);
			const menu = page.getByTestId('viewer-settings-menu');
			const other = slot(page, b);

			await expect(other, 'the other slot starts expanded').toHaveAttribute('class', /^((?!collapsed).)*$/);
			await slot(page, a).getByTestId('viewer-settings-cog').click();
			await expect(menu).toBeVisible();

			// The other node's slot header — the exact element whose `stopPropagation` keeps the dismiss
			// pointerdown off `window`, and whose click collapses its viewer.
			await clickAt(page, other.locator('header'));
			await expect(menu, 'the click dismissed the menu').toBeHidden();
			await expect(
				other,
				'and was swallowed — the slot it landed on is untouched'
			).toHaveAttribute('class', /^((?!collapsed).)*$/);
		});

		test('a click on the configured slot dismisses the menu without collapsing it', async ({
			page
		}) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await oscillator(page, [40, 40]);
			const menu = page.getByTestId('viewer-settings-menu');
			const own = slot(page, uid);

			await own.getByTestId('viewer-settings-cog').click();
			await expect(menu).toBeVisible();
			await clickAt(page, own.locator('header'));
			await expect(menu, 'the click dismissed the menu').toBeHidden();
			await expect(
				own,
				'the viewer being configured is not collapsed by its own dismiss-click'
			).toHaveAttribute('class', /^((?!collapsed).)*$/);
		});

		/* Escape is a DISMISSAL, and a dismissal is consumed by the surface that owns it.
		 *
		 * `Popover` took Escape at bubble phase and called `onDismiss()` without stopping it, so on the
		 * canvas host the press reached `NodeEditorPanel`'s own window listener too — and that handler
		 * cannot exclude it, because the trigger is slot-header chrome, so the panel is still active and
		 * the press targets a <button> that is in no tag allowlist and inside no open <dialog>. One
		 * Escape therefore dismissed the menu AND cleared the canvas selection, or — inside a sub-patch,
		 * with nothing selected — silently popped one level.
		 *
		 * `ContextMenu` was fixed out of exactly this at 5a7f468 and carries the reasoning in place;
		 * `Popover` kept the bug. Both now take the key at window-CAPTURE, which runs before every
		 * window-bubble listener, and consume it. Safe inside the surface: neither `TextInput` nor
		 * `NumberInput` binds Escape. */
		test('Escape dismisses the menu without also reaching the canvas under it', async ({ page }) => {
			await page.goto('/');
			await waitForApp(page);
			const uid = await oscillator(page, [40, 40]);
			const menu = page.getByTestId('viewer-settings-menu');
			const selectedCount = async (): Promise<number> =>
				(await page.evaluate(() => (window as any).goofi.query.selection())).nodes.length;

			// Select the node, so the editor's Escape branch has something to destroy.
			await page
				.locator('.goofi-node')
				.filter({ has: page.locator(`.slot-viewer[data-node="${uid}"]`) })
				.locator('.header')
				.first()
				.click();
			await expect.poll(selectedCount, { message: 'the node is selected' }).toBe(1);

			await slot(page, uid).getByTestId('viewer-settings-cog').click();
			await expect(menu).toBeVisible();
			await page.keyboard.press('Escape');

			await expect(menu, 'Escape dismisses the popover').toBeHidden();
			expect(await selectedCount(), 'and is consumed — the selection under it survives').toBe(1);
		});
	});
});

test.describe('the viewer hover readout', () => {
	/**
	 * The fine-pointer half of `touch-viewer.spec.ts`, over the same list of surfaces.
	 *
	 * Gating the viewer's five `:hover` rules on `(hover: hover)` is a change a mouse must not be able
	 * to see, and "the existing specs still pass" does not prove that: none of them reads a hover
	 * colour. This does — every surface the coarse spec proves stays at rest is proved here to still
	 * move.
	 */
	test('every viewer hover rule still answers a mouse', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uid = await addNode(page, 'Oscillator', 'inputs', [40, 40]);
		await waitForNode(page, uid);
		const slot = (p: Page) => p.locator(`.slot-viewer[data-node="${uid}"]`);
		await expect(slot(page)).toBeVisible();
		try {
			await unhover(page);
			const rest = new Map<string, string[]>();
			for (const s of VIEWER_HOVER_SURFACES) rest.set(s.name, await surfaceStyles(page, uid, s));

			for (const s of VIEWER_HOVER_SURFACES) {
				await hoverSettled(page, uid, s);
				const now = await surfaceStyles(page, uid, s);
				expect(now, `${s.name} still lights up under the mouse`).not.toEqual(rest.get(s.name));
				await unhover(page);
			}
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNode(u), uid);
			await waitForNoNode(page, uid).catch(() => {});
		}
	});
});

test.describe('the viewer frame cap', () => {
	/**
	 * The viewer paint cap (Phil, 2026-08-08): the frontend paints at most `MAX_VIEWER_FPS` (30),
	 * app-wide, however fast the producer runs or the display refreshes. Nodes update at ≤30 Hz in
	 * practice; an uncapped rAF flush painted at display rate — 120 fps on a high-refresh phone —
	 * re-painting frames that carried no new update.
	 *
	 * Driven through the real pipeline: Oscillators with their viewers subscribed, and the perf HUD
	 * read after the meter settles. The bound is generous (≤ 40 against a 30 cap) because the HUD's
	 * meter is a 500ms window — but an UNCAPPED flush in this harness paints at ~60 (headless vsync),
	 * so the bound still separates the two behaviours cleanly.
	 *
	 * Two axes, and the second one is why this file has a second node. The cap is app-wide: ONE rAF
	 * flush repaints every stream that has a new frame, so the readout must not move with the number
	 * of open streams either. Asserting that needed a TWO-stream fixture — with one node the paint
	 * rate and the old per-slot SUM are the same number, so this spec sat green for two days while the
	 * HUD climbed 30 fps per node added (`fps-counter-investigation.md`). A single-stream fixture
	 * cannot express a summing bug.
	 */
	const readFps = (page: import('@playwright/test').Page): Promise<number> =>
		page
			.getByTestId('perf-hud')
			.locator('.fps')
			.evaluate((el) => parseFloat(el.textContent ?? '0'));

	test('the viewer paint rate stays at the cap, however fast the producer runs', async ({ page }) => {
		await page.goto('/');
		await waitForApp(page);
		const uids = [await addNode(page, 'Oscillator', 'inputs', [60, 60])];
		try {
			await waitForNode(page, uids[0]);
			await page.evaluate(
				(u) => (window as any).goofi.commands.updateParam(u, 'common', 'max_frequency', 120),
				uids[0]
			);
			// Let the new rate flow and the HUD's 500ms window fill.
			await expect(page.getByTestId('perf-hud')).toBeAttached();
			await page.waitForTimeout(1500);
			const one = await readFps(page);
			expect(one, 'frames flow at all (the producer really runs)').toBeGreaterThan(15);
			expect(one, 'painted at the cap, not at the display or producer rate').toBeLessThanOrEqual(40);

			// The node-count axis: a second stream, painted by the same app-wide flush.
			uids.push(await addNode(page, 'Oscillator', 'inputs', [60, 300]));
			await waitForNode(page, uids[1]);
			await page.waitForTimeout(1500);
			const two = await readFps(page);
			expect(two, 'the second stream really paints too').toBeGreaterThan(15);
			expect(
				two,
				`the paint rate is one app-wide number, not a per-stream sum (1 node: ${one}, 2 nodes: ${two})`
			).toBeLessThanOrEqual(40);
		} finally {
			await page.evaluate((u) => (window as any).goofi.commands.removeNodes(u), uids);
		}
	});
});

test.describe('the WebGL image viewer', () => {
	/**
	 * The GL image path's texture-format contract, checked against a REAL browser.
	 *
	 * `glSupports` (frontend/src/lib/viewers/imageGL.ts) routes float RGB to the GPU on the grounds
	 * that RGB32F is a required sized internal format for textures in WebGL2 — it is only
	 * non-renderable and non-filterable, neither of which the renderer needs. That claim is a
	 * platform fact a unit test cannot check, and getting it wrong is expensive: RGB is HD video, and
	 * the fallback is a w*h*3-iteration JS loop on the main thread. So exercise the exact upload the
	 * renderer performs and assert the driver accepted it.
	 */
	test('the GL image path can upload float grayscale, RGB and RGBA textures', async ({ page }) => {
		await page.goto('/');
		const r = await page.evaluate(() => {
			const gl = document.createElement('canvas').getContext('webgl2');
			if (!gl) return null;
			gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
			gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
			// One 2x2 upload per channel count, mirroring GLImageRenderer.render's format ladder.
			const upload = (internal: number, format: number, c: number) => {
				gl.texImage2D(gl.TEXTURE_2D, 0, internal, 2, 2, 0, format, gl.FLOAT, new Float32Array(4 * c));
				return gl.getError();
			};
			const gray = upload(gl.R32F, gl.RED, 1);
			const rgb = upload(gl.RGB32F, gl.RGB, 3);
			const rgba = upload(gl.RGBA32F, gl.RGBA, 4);
			// NEAREST is always legal for a float texture; the renderer only asks for LINEAR when
			// OES_texture_float_linear is present.
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
			return { gray, rgb, rgba, filter: gl.getError(), NO_ERROR: gl.NO_ERROR };
		});
		expect(r, 'WebGL2 must be available in the test browser').not.toBeNull();
		expect(
			[r!.gray, r!.rgb, r!.rgba, r!.filter],
			'every float texture format the renderer uploads must be accepted'
		).toEqual([r!.NO_ERROR, r!.NO_ERROR, r!.NO_ERROR, r!.NO_ERROR]);
	});
});
