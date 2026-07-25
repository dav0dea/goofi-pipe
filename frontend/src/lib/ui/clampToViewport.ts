/**
 * `clampToViewport` — the one correct anchored-overlay clamp (spec §3), the SSOT both the `Popover`
 * primitive and `ContextMenu` position against. Lifted verbatim from `ContextMenu.svelte`'s measured
 * viewport math and generalised from a spawn point to an anchor rect: a popover opens flush under the
 * anchor's bottom-left, then shifts back on-screen — never past a small viewport `MARGIN` — if it
 * would overflow the right or bottom edge. A point anchor is just the degenerate rect
 * (`left == right`, `top == bottom`, zero-sized), which is how ContextMenu spends it. Pure, so it is
 * unit-tested without a DOM (the component just feeds it real `getBoundingClientRect()` measurements).
 *
 * The shift (not a flip) mirrors ContextMenu exactly; `Math.max(MARGIN, …)` floors an
 * overflowing/oversized menu to the margin so it is never pushed off-screen (never negative).
 */

/** The gap kept between a clamped popover and the viewport edge — ContextMenu's 6px. */
const MARGIN = 6;

/** The subset of a `DOMRect` the clamp reads — a real `getBoundingClientRect()` satisfies it. */
export type AnchorRect = Pick<DOMRect, 'left' | 'top' | 'right' | 'bottom' | 'width' | 'height'>;

export interface Size {
	width: number;
	height: number;
}

/** The clamped top-left, in viewport (fixed) coordinates, ready for `left`/`top`. */
export interface Placement {
	left: number;
	top: number;
}

export function clampToViewport(anchor: AnchorRect, menu: Size, viewport: Size): Placement {
	// Preferred origin: flush under the anchor's bottom-left (the popover hangs below its trigger).
	let left = anchor.left;
	let top = anchor.bottom;
	if (left + menu.width > viewport.width - MARGIN) {
		left = Math.max(MARGIN, viewport.width - menu.width - MARGIN);
	}
	if (top + menu.height > viewport.height - MARGIN) {
		top = Math.max(MARGIN, viewport.height - menu.height - MARGIN);
	}
	return { left, top };
}
