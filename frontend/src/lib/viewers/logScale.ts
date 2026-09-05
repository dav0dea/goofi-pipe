/**
 * The window a log scale may be given. uPlot reads its tick increment off `log(min)`, so a bound
 * at or below zero hands it -Infinity or NaN and its generator pushes until the array length
 * throws — a frozen tab, then a dead renderer. A PSD reaches zero, so this is the ordinary case.
 */
export function logSafe(min: number, max: number): [number, number] {
	const hi = Number.isFinite(max) && max > 0 ? max : 1;
	const lo = Number.isFinite(min) && min > 0 && min < hi ? min : hi / 1e3;
	return [lo, hi];
}
