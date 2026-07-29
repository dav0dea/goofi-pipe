import type { Page } from '@playwright/test';

/**
 * Gap in px between the node editor's on-canvas control cluster and its panel's bottom-left
 * corner, plus the live root font-size to measure it against (`--space-*` is rem and the `html`
 * base is a responsive clamp, so the expected number moves with the viewport by design).
 *
 * Shared by the fine-pointer and coarse-pointer specs, which assert different rungs of the same
 * rule — and shared through `lib/` rather than by importing one spec from the other, since loading
 * a spec file is how Playwright registers its tests.
 */
export function controlsInset(
	page: Page
): Promise<{ left: number; bottom: number; rem: number }> {
	return page.locator('.canvas-wrap').first().evaluate((wrap) => {
		const w = wrap.getBoundingClientRect();
		const c = wrap.querySelector('.svelte-flow__controls')!.getBoundingClientRect();
		return {
			left: c.left - w.left,
			bottom: w.bottom - c.bottom,
			rem: parseFloat(getComputedStyle(document.documentElement).fontSize)
		};
	});
}
