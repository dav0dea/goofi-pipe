import { test, expect } from '@playwright/test';
import { waitForApp } from '../lib/app';
import { controlsInset } from '../lib/editor';

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
