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
 * boundary width. It reads cached intrinsic widths and a budget its caller measures off the OTHER
 * boxes in the header, so the second pass computes the first pass's answer and writes nothing.
 *
 * "The other boxes" is not "boxes that cannot move" — the status cluster shrinks (see TopBar's own
 * note), so the budget is re-measured rather than assumed constant. What makes it converge is that
 * the only growable box is the tab strip: a spilled action's width goes there 1:1, leaving the fit
 * condition a monotone threshold in the container width.
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
	/** The overflow trigger's own width, in px. */
	trigger: number;
	/**
	 * Is the trigger on screen even when NOTHING has spilled? Default `true`.
	 *
	 * The app header's is: its menu carries the canvas commands at every width, so the button is
	 * resident chrome and its width is simply always in the budget. The panel header's is not — its
	 * menu holds the spilled actions and nothing else, so at a width where all of them fit there is
	 * no menu to open and drawing a `⋯` beside them would be a door onto an empty room.
	 *
	 * Saying so is what keeps the plan HONEST rather than merely conservative: charged
	 * unconditionally, a non-resident bar spills its first item at the width where everything would
	 * still have fit without the trigger it is being charged for.
	 */
	residentTrigger?: boolean;
}

/**
 * The ids that must move into the overflow menu, given the budget.
 *
 * `spillOrder` is the priority order LOWEST FIRST: the id named first is the first the bar gives
 * up. Ids it names that are not in `items` are ignored, so a caller may keep one constant order.
 *
 * `trigger` is charged whenever it is on screen — which for the app header is unconditionally,
 * because that menu is RESIDENT chrome: it carries the canvas commands (delete, group, select-all,
 * copy/paste/duplicate, multi-select) at every width, since those have no bar slot to lose. So
 * there the second trap D-R6 names — a trigger that appears and pushes the last item out — is
 * designed away rather than guarded against. A bar whose menu holds only the spilled items says
 * `residentTrigger: false`, and pays that trap with the property below instead.
 *
 * What survives either way is MONOTONICITY: the answer is the first prefix of `spillOrder` that
 * fits, and a prefix that fits at one budget still fits at a larger one — so a wider bar can never
 * spill more. That is the whole non-oscillation argument, and it does not depend on the trigger
 * being free.
 *
 * Deliberately NOT implementing D-R6's "never leave exactly one item in the overflow" hint: the
 * first thing to spill is the Save caret, and losing it alone degrades cleanly (the split control
 * becomes a plain Save button). Forcing `Load…` out alongside it would remove a reachable button
 * to satisfy a symmetry nobody sees.
 */
export function planOverflow(
	items: OverflowItem[],
	spillOrder: string[],
	{ gap, budget, trigger, residentTrigger = true }: FitOpts
): Set<string> {
	const spilled = new Set<string>();
	// Counted as BOXES rather than as items plus a constant: n boxes are separated by n − 1 gaps,
	// and whether the trigger is one of them is exactly what `residentTrigger` decides.
	const fits = (): boolean => {
		const boxes = items.filter((it) => !spilled.has(it.id)).map((it) => it.width);
		if (residentTrigger || spilled.size > 0) boxes.push(trigger);
		const used = boxes.reduce((a, w) => a + w, 0) + gap * Math.max(0, boxes.length - 1);
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
