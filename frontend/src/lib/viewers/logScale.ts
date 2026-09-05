/**
 * The window a log scale may be given: positive and finite, or `log10` makes the scale NaN. A PSD
 * reaches zero, so this is the ordinary case.
 */
export function logSafe(min: number, max: number): [number, number] {
	const hi = Number.isFinite(max) && max > 0 ? max : 1;
	const lo = Number.isFinite(min) && min > 0 && min < hi ? min : hi / 1e3;
	return [lo, hi];
}

/**
 * The grid lines of a log axis: one per decade, from the decade holding `min` to the one holding
 * `max`. uPlot's own walk steps by an increment it looks up in a table, and a minimum in the top
 * of a decade or below 1e-22 steps off that table and never terminates.
 */
export function logSplits(min: number, max: number): number[] {
	const [lo, hi] = logSafe(min, max);
	const first = Math.floor(Math.log10(lo));
	const last = Math.ceil(Math.log10(hi));
	const out: number[] = [];
	for (let k = first; k <= last; k++) out.push(Number(`1e${k}`));
	return out;
}
