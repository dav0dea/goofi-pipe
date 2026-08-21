/** These mirror `.row`'s CSS and stay px: the estimate is computed before layout exists. */
export const LINE_H = 16;
export const PAD = 4;
export const BORDER = 1;
export const COLLAPSE_LINES = 3;

/** A row's height before it has been measured; `floor` is the row's content floor in px. */
export function estimateRowHeight(lines: number, expanded: boolean, floor = 0): number {
	const shown = expanded ? lines : Math.min(lines, COLLAPSE_LINES);
	return Math.max(shown * LINE_H, floor) + PAD + BORDER;
}
