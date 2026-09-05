/** What a vertical drag of `dy` pixels does to a value in `[min, max]`, on `step`. A full sweep is
 * `span` pixels, so a taller drag is not a wilder jump — the range decides, not the screen. */
export function turnedBy(
	value: number,
	dy: number,
	min: number,
	max: number,
	step: number,
	span = 160
): number {
	const next = value - (dy / span) * (max - min);
	const stepped = step > 0 ? Math.round(next / step) * step : next;
	const clamped = Math.min(max, Math.max(min, stepped));
	// Float arithmetic on a step of 0.01 leaves a tail; the step says what precision is meant.
	const places = step > 0 && step < 1 ? Math.ceil(-Math.log10(step)) : 0;
	return Number(clamped.toFixed(places));
}
