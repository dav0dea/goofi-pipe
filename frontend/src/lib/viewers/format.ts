/** Compact numeric label for a tick or bound; non-finite → '' so no NaN ever renders. */
export function formatTick(v: number): string {
	if (!Number.isFinite(v)) return '';
	const abs = Math.abs(v);
	if (abs === 0) return '0';
	if (abs >= 10000 || abs < 0.01) return v.toExponential(1);
	if (abs >= 100) return v.toFixed(0);
	if (abs >= 1) return v.toFixed(2);
	return v.toFixed(3);
}
