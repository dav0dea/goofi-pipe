/**
 * The primitive galleries under a coarse pointer. A gallery route mounts no AppShell, opens no
 * socket and names no patch, so these are isolated by construction and run fully parallel.
 */

import { test, expect, type Locator } from '@playwright/test';
import { SAMPLES, controlLocator } from '../lib/uiSweep';

test.describe('every $lib/ui primitive meets the coarse floor', () => {
	// Runs ONLY under the `touch` project (Pixel 7 emulation → hasTouch+isMobile flip
	// (pointer:coarse)/(hover:none) true), so app.css floors --hit to 44px. The IconButton's
	// rendered box must therefore be a real 44px tap target in BOTH dimensions while its glyph
	// element stays visually small — the deferred-from-F icon-control hit-targeting requirement.
	test('IconButton meets the 44px coarse tap target while its glyph stays small', async ({ page }) => {
		await page.goto('/dev/ui');
		const btn = page.getByTestId('ui-icon-primary-md');
		await btn.waitFor();

		const box = await btn.boundingBox();
		expect(box, 'the IconButton has a rendered box').not.toBeNull();
		expect(box!.width, 'coarse tap-target width >= 44').toBeGreaterThanOrEqual(44);
		expect(box!.height, 'coarse tap-target height >= 44').toBeGreaterThanOrEqual(44);

		// The glyph itself must NOT be inflated to the tap target — it stays small and centered.
		const glyph = btn.locator('.glyph');
		const gbox = await glyph.boundingBox();
		expect(gbox, 'the glyph has a rendered box').not.toBeNull();
		expect(gbox!.width, 'the glyph stays small (not the tap target)').toBeLessThan(24);
		expect(gbox!.height, 'the glyph stays small (not the tap target)').toBeLessThan(24);
	});

	// The Field-family controls must also honour the coarse --hit floor: a NumberInput inherits the
	// app.css `min-height: var(--hit)` (44px under a coarse pointer), and the Toggle sizes its box to
	// --hit. Both must be real tap targets on touch — the Field family is what R renders on phones.
	test('Field controls meet the 44px coarse tap target', async ({ page }) => {
		await page.goto('/dev/ui');

		const num = page.getByTestId('ui-field-number');
		await num.waitFor();
		const nbox = await num.boundingBox();
		expect(nbox, 'the NumberInput has a rendered box').not.toBeNull();
		expect(nbox!.height, 'NumberInput height >= 44 under coarse').toBeGreaterThanOrEqual(44);

		const toggle = page.getByTestId('ui-toggle');
		await toggle.waitFor();
		const tbox = await toggle.boundingBox();
		expect(tbox, 'the Toggle has a rendered box').not.toBeNull();
		expect(tbox!.height, 'Toggle height >= 44 under coarse').toBeGreaterThanOrEqual(44);

		// Width too: the painted track stays a switch-sized 2.4rem (< 44px under the coarse html base), so
		// the coarse floor is met by the `::after` hit-rect overlay (mirroring IconButton), NOT by widening
		// the box. Measure the effective tap target: the element box grown by the pseudo's negative insets.
		const tap = await toggle.evaluate((el) => {
			const r = el.getBoundingClientRect();
			const cs = getComputedStyle(el, '::after');
			const px = (v: string) => parseFloat(v) || 0;
			return { width: r.width - px(cs.left) - px(cs.right), height: r.height - px(cs.top) - px(cs.bottom) };
		});
		// Round to the nearest device pixel: the calc()-derived rect resolves to exactly --hit in math but
		// getBoundingClientRect/getComputedStyle report sub-pixel noise (e.g. 43.99999). Pre-fix ≈33.6 → 34,
		// still well under the floor, so the RED guard holds.
		expect(Math.round(tap.width), 'Toggle coarse tap-target width >= 44 (via the ::after hit-rect)').toBeGreaterThanOrEqual(44);
		expect(Math.round(tap.height), 'Toggle coarse tap-target height >= 44').toBeGreaterThanOrEqual(44);
	});

	// Tabs and Disclosure are chrome R renders on phones — each tab and the disclosure summary must be a
	// real 44px tap target under a coarse pointer (both size their control to var(--hit)).
	test('Tabs + Disclosure meet the 44px coarse tap target', async ({ page }) => {
		await page.goto('/dev/ui');

		const tab = page.getByTestId('ui-tabs').getByRole('tab').first();
		await tab.waitFor();
		const tabBox = await tab.boundingBox();
		expect(tabBox, 'the tab has a rendered box').not.toBeNull();
		expect(tabBox!.height, 'tab height >= 44 under coarse').toBeGreaterThanOrEqual(44);

		const summary = page.getByTestId('ui-disclosure').getByRole('button');
		await summary.waitFor();
		const sbox = await summary.boundingBox();
		expect(sbox, 'the disclosure summary has a rendered box').not.toBeNull();
		expect(sbox!.height, 'disclosure summary height >= 44 under coarse').toBeGreaterThanOrEqual(44);
	});

	// The whole-library touch roll-up (Task 8). The named tests above spot-check a spread; this one
	// enumerates EVERY interactive primitive off the shared registry (pinned to `$lib/ui`'s barrel by the
	// default-project sweep) and asserts each one's real control meets the coarse 44px floor — so a future
	// interactive primitive is held to the touch floor automatically, not only if someone remembers to add
	// a case here. The control is the actual tappable element (a tab, a Disclosure summary, the range/select
	// inside a wrapper), not the sample wrapper.
	test('every interactive primitive meets the 44px coarse tap target', async ({ page }) => {
		await page.goto('/dev/ui');
		for (const [name, sample] of Object.entries(SAMPLES)) {
			if (!sample.interactive) continue;
			const control = controlLocator(page, sample);
			await control.first().waitFor();
			const box = await control.first().boundingBox();
			expect(box, `${name} has a rendered control box`).not.toBeNull();
			expect(box!.height, `${name} coarse tap-target height >= 44`).toBeGreaterThanOrEqual(44);
		}
	});
});

test.describe('the inspector gallery clears the focus-zoom threshold', () => {
	// Runs ONLY under the `touch` project (Pixel 7 → (pointer:coarse)/(hover:none) true), where app.css
	// floors input/select/textarea font-size to 16px to defeat iOS focus-zoom. Two inspector controls
	// out-specified that floor with a sub-16px F token — the fx multi-line editor (--fs-micro ≈ 10px) and
	// the header rename input (--fs-strong ≈ 14px on phone) — so a tap into either force-zoomed the
	// viewport on focus. Each must now resolve to >= 16px under a coarse pointer (the desktop sub-16px
	// sizes are kept — only the coarse-pointer override lifts them).
	const fontPx = (loc: Locator): Promise<number> =>
		loc.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));

	test('the fx multi-line editor is >= 16px under a coarse pointer (no iOS focus-zoom)', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		const field = page.getByTestId('inspector-fx');
		// Enable fx, then grow the in-panel multi-line editor so the textarea renders.
		await field.getByTestId('param-fx-toggle').click();
		await field.getByTestId('param-expr-expand').click();
		const ta = field.getByTestId('param-expr-multiline');
		await expect(ta).toBeVisible();
		expect(await fontPx(ta), 'textarea font-size >= 16 defeats iOS focus-zoom').toBeGreaterThanOrEqual(16);
	});

	test('the header rename input is >= 16px under a coarse pointer (no iOS focus-zoom)', async ({
		page
	}) => {
		await page.goto('/dev/inspector');
		// The ParamForm sample renders its identity header; open the inline rename to mount its input.
		const form = page.getByTestId('inspector-form');
		await form.getByTestId('node-name').click();
		const input = form.getByTestId('node-name-input');
		await expect(input).toBeVisible();
		expect(await fontPx(input), 'rename input font-size >= 16 defeats iOS focus-zoom').toBeGreaterThanOrEqual(16);
	});
});
