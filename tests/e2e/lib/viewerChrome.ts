import { expect, type Page } from '@playwright/test';

/**
 * The slot viewer's hover-feedback surfaces — every `:hover` rule the in-canvas viewer chrome
 * carries, as data.
 *
 * Two specs ask opposite questions of this one list: `viewer-hover.spec.ts` proves a fine pointer
 * still gets every one of them, `touch-viewer.spec.ts` proves a coarse pointer gets none. Shared
 * through `lib/` rather than by importing one spec from the other, since loading a spec file is how
 * Playwright registers its tests — and a second copy is a second thing to keep true.
 */
export interface HoverSurface {
	/** What the assertion message calls it. */
	name: string;
	/** Selector, relative to the slot viewer of the node under test. */
	sel: string;
	/** The computed properties the hover rule moves. */
	props: string[];
}

export const VIEWER_HOVER_SURFACES: HoverSurface[] = [
	{ name: 'the header bar', sel: 'header', props: ['background-color'] },
	{ name: 'the disclosure triangle', sel: '.tri svg', props: ['fill'] },
	{ name: 'the slot name', sel: '[data-testid="slot-output"]', props: ['background-color'] },
	{ name: 'the viewer-kind select', sel: '[data-testid="viewer-kind"] select', props: ['color', 'border-top-color'] },
	{ name: 'the settings cog', sel: '[data-testid="viewer-settings-cog"]', props: ['color'] }
];

const locate = (page: Page, uid: string, s: HoverSurface) =>
	page.locator(`.slot-viewer[data-node="${uid}"] ${s.sel}`).first();

/** One surface's computed values, in the order `props` names them. */
export async function surfaceStyles(page: Page, uid: string, s: HoverSurface): Promise<string[]> {
	return locate(page, uid, s).evaluate(
		(el, props) => props.map((p) => getComputedStyle(el).getPropertyValue(p)),
		s.props
	);
}

/** Park the mouse well clear of the node, so a resting reading really is at rest. */
export async function unhover(page: Page): Promise<void> {
	await page.mouse.move(2, 2);
}

/**
 * Put the mouse on a surface and let the paint settle.
 *
 * The settle is the whole point: every one of these transitions on `--dur-fast`, so an assertion
 * fired the instant the pointer lands reads the value it is leaving — which greens a "stays at
 * rest" check for the wrong reason. 300ms clears `--dur-slow` (200ms) with room to spare, and the
 * `toHaveCSS` that follows still retries.
 */
export async function hoverSettled(page: Page, uid: string, s: HoverSurface): Promise<void> {
	const box = await locate(page, uid, s).boundingBox();
	expect(box, `${s.name} is on screen`).toBeTruthy();
	await page.mouse.move(Math.round(box!.x + box!.width / 2), Math.round(box!.y + box!.height / 2));
	await page.waitForTimeout(300);
}
