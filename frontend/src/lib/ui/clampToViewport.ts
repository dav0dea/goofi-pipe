/**
 * `clampToViewport` — the one correct anchored-overlay clamp (spec §3), the SSOT both the `Popover`
 * primitive and `ContextMenu` position against. Lifted verbatim from `ContextMenu.svelte`'s measured
 * viewport math and generalised from a spawn point to an anchor rect: a popover opens flush under the
 * anchor's bottom-left, then shifts back on-screen — never past a small viewport `MARGIN` on ANY
 * edge — if it would overflow. A point anchor is just the degenerate rect
 * (`left == right`, `top == bottom`, zero-sized), which is how ContextMenu spends it. Pure, so it is
 * unit-tested without a DOM (the component just feeds it real `getBoundingClientRect()` measurements).
 *
 * The shift (not a flip) mirrors ContextMenu exactly; `Math.max(MARGIN, …)` floors an
 * overflowing/oversized menu to the margin so it is never pushed off-screen (never negative).
 * A caller whose anchor sits at the BOTTOM of the screen opts into `{ flip: true }` — see below.
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

export interface ClampOptions {
	/**
	 * Allow the popover to flip ABOVE the anchor when it does not fit below but does fit above.
	 * Opt-in, because the shift is wrong only for a BOTTOM-anchored trigger: shifting slides the
	 * surface up until it fits and then encloses the very trigger that opened it (the floating error
	 * chip), making the toggle-to-close unreachable. A point anchor (ContextMenu) has no trigger to
	 * enclose, so it keeps the plain shift.
	 */
	flip?: boolean;
}

export function clampToViewport(
	anchor: AnchorRect,
	menu: Size,
	viewport: Size,
	{ flip = false }: ClampOptions = {}
): Placement {
	// Preferred origin: flush under the anchor's bottom-left (the popover hangs below its trigger).
	let left = anchor.left;
	let top = anchor.bottom;
	if (left + menu.width > viewport.width - MARGIN) {
		left = Math.max(MARGIN, viewport.width - menu.width - MARGIN);
	}
	if (top + menu.height > viewport.height - MARGIN) {
		// Above the anchor, keeping the same MARGIN as a gap — taken only when it lands fully
		// on-screen under the module's own margin rule, else the shift below stands.
		const above = anchor.top - menu.height - MARGIN;
		top = flip && above >= MARGIN ? above : Math.max(MARGIN, viewport.height - menu.height - MARGIN);
	}
	// The near edges get the same floor the far ones already have. An anchor rect can start off-screen
	// (a caller that aligns the menu's RIGHT edge or its centre to a point near x = 0 computes a
	// negative origin), and "clamp to viewport" has to mean all four edges or it means nothing.
	return { left: Math.max(MARGIN, left), top: Math.max(MARGIN, top) };
}

/**
 * The `viewport` both callers feed the clamp: layout WIDTH, but VISUAL HEIGHT. A soft keyboard
 * covers the bottom of the screen without shrinking `innerHeight`, so an overlay clamped to the
 * layout height is parked underneath the very keyboard that raised it.
 *
 * The overlap is `--kb-inset`, published on <html> by `stores/device.svelte.ts` — the app's one
 * subscriber to `visualViewport`. It is READ FROM THE DOM here rather than imported, so `$lib/ui`
 * stays free of app stores: a primitive that reaches up into the store layer pulls that layer into
 * the primitives' bundle chunk, and the reshuffled CSS emission order costs the app a flash of
 * unstyled chrome on first paint (measured — it reddened `inspector-gallery.spec.ts`). The published
 * custom property is the seam; both overlays measure through this one function so they cannot
 * disagree. The measurement lives beside the pure clamp for that reason, and only that reason.
 */
export function overlayViewport(): Size {
	const inset =
		parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--kb-inset')) || 0;
	return { width: window.innerWidth, height: window.innerHeight - inset };
}
