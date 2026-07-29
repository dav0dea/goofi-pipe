/**
 * Progressive overflow — the measured half of D-R6.
 *
 * The app header's actions live in the bar and give themselves up to the overflow menu ONE AT A
 * TIME, lowest priority first, as the width runs out. That is not a breakpoint and it cannot be
 * written in CSS: "does item N still fit" is a question about intrinsic widths, which no media or
 * container query can ask. So the decision is arithmetic, and it lives here — pure, away from the
 * DOM, where it can be pinned by unit tests rather than by a screenshot.
 *
 * The one property that matters is that `planOverflow` is a function of its ARGUMENTS. Moving an
 * item out of the bar changes what is rendered, which re-fires the ResizeObserver that decided to
 * move it; if the plan read the bar's own content it would flip-flop forever at exactly the
 * boundary width. It reads cached intrinsic widths and a budget derived from boxes the plan
 * cannot move, so the second pass computes the first pass's answer and writes nothing.
 */

/** One measurable action in the bar, in DOM order. */
export interface OverflowItem {
	/** Stable id — the action's `data-testid`. */
	id: string;
	/** Intrinsic width in px, measured with every item visible. */
	width: number;
}

export interface FitOpts {
	/** The bar's inter-item gap, in px. */
	gap: number;
	/** Horizontal space the actions may occupy, in px. */
	budget: number;
	/** The overflow trigger's own width, in px — always charged (see below). */
	trigger: number;
}

/**
 * The ids that must move into the overflow menu, given the budget.
 *
 * `spillOrder` is the priority order LOWEST FIRST: the id named first is the first the bar gives
 * up. Ids it names that are not in `items` are ignored, so a caller may keep one constant order.
 *
 * `trigger` is charged unconditionally, because the menu is RESIDENT chrome — it carries the
 * canvas commands (delete, group, select-all, copy/paste/duplicate, multi-select) at every width,
 * since those have no bar slot to lose. So there is no width at which the trigger appears and
 * pushes the last item out, which is the second trap D-R6 names; it is designed away rather than
 * guarded against.
 *
 * Deliberately NOT implementing D-R6's "never leave exactly one item in the overflow" hint: the
 * first thing to spill is the Save caret, and losing it alone degrades cleanly (the split control
 * becomes a plain Save button). Forcing `Load…` out alongside it would remove a reachable button
 * to satisfy a symmetry nobody sees.
 */
export function planOverflow(
	items: OverflowItem[],
	spillOrder: string[],
	{ gap, budget, trigger }: FitOpts
): Set<string> {
	const spilled = new Set<string>();
	// n visible items sit in n+1 boxes with the trigger, so they are separated by n gaps.
	const fits = (): boolean => {
		let used = trigger;
		for (const it of items) if (!spilled.has(it.id)) used += it.width + gap;
		return used <= budget;
	};
	const known = new Set(items.map((i) => i.id));
	for (const id of spillOrder) {
		if (fits()) return spilled;
		if (known.has(id)) spilled.add(id);
	}
	return spilled;
}

/**
 * Intrinsic widths, measured once and re-measured when the root font size moves.
 *
 * The `html` base is a responsive `clamp()` (and the coarse-pointer floor raises it again), so the
 * same button is a different number of pixels at a different root size — the third trap. Keyed on
 * the root size rather than on a resize, because a resize that does not cross a clamp step leaves
 * every width exactly where it was and re-measuring would only cost a forced layout.
 */
export interface WidthCache {
	/** Widths at `rem`, measuring only when the last measurement was taken at another root size. */
	widths(rem: number): number[];
	/** Drop the cache — for a change the root size does not describe (an item's own label moved). */
	invalidate(): void;
}

export function createWidthCache(measure: () => number[]): WidthCache {
	let cached: number[] | null = null;
	let at = 0;
	return {
		widths(rem: number): number[] {
			if (!cached || rem !== at) {
				cached = measure();
				at = rem;
			}
			return cached;
		},
		invalidate(): void {
			cached = null;
		}
	};
}
