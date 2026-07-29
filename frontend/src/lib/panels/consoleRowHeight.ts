/**
 * The console's virtual-scroller row-height model. Pure and out here rather than inside the
 * component, because it is the piece with real arithmetic: `layout.cum` sums this ESTIMATE for
 * every row the panel's ResizeObserver has not measured yet — on a full ring buffer, most of them
 * — so when the model and the DOM disagree the scrollbar reads the wrong length and the content
 * drifts under the thumb as rows measure in.
 *
 * The three constants mirror `.row`'s CSS, which says so at each declaration. They stay px (not
 * `--space-*`/rem) for exactly that reason: the estimate is computed in px before layout exists.
 */

/** px per text line — `line-height: 16px` on the row's children. */
export const LINE_H = 16;
/** the row's vertical padding — `.row { padding: 2px … }`, both sides. */
export const PAD = 4;
/** the row's own bottom hairline: part of its box, so part of the estimate. */
export const BORDER = 1;
/** lines shown before a row collapses. */
export const COLLAPSE_LINES = 3;

/**
 * A row's height before it has been measured. `floor` is the row's CONTENT floor: under a coarse
 * pointer every control the row hosts is `--hit` tall, so the row is too while its text is still
 * one line (C16). 0 — the fine-pointer answer — leaves the text alone deciding.
 */
export function estimateRowHeight(lines: number, expanded: boolean, floor = 0): number {
	const shown = expanded ? lines : Math.min(lines, COLLAPSE_LINES);
	return Math.max(shown * LINE_H, floor) + PAD + BORDER;
}
