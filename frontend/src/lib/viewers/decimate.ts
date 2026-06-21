/**
 * Min/max decimation for line plots (report A15).
 *
 * When a channel has far more samples than the canvas has pixel columns,
 * handing every sample to uPlot wastes time drawing sub-pixel detail. This
 * reduces each channel to ~2 points per target column — the per-bucket min and
 * max — which is the standard oscilloscope technique: it preserves spikes (the
 * bucket max/min survive) while cutting the point count to the display budget.
 *
 * All channels share one x grid (two x slots per bucket) so the result is valid
 * uPlot AlignedData. Pure and dependency-free, so it's unit-testable.
 */
export interface Decimated {
	xs: number[];
	ys: number[][];
}

export function decimateMinMax(
	channels: ArrayLike<number>[],
	m: number,
	targetCols: number
): Decimated {
	const buckets = Math.max(1, Math.min(targetCols, m));
	const bucketSize = m / buckets;
	const xs = new Array<number>(buckets * 2);
	const ys = channels.map(() => new Array<number>(buckets * 2));

	for (let b = 0; b < buckets; b++) {
		const start = Math.floor(b * bucketSize);
		const end = Math.min(m, Math.floor((b + 1) * bucketSize));
		// Two distinct x slots per bucket so min and max are separate points.
		xs[b * 2] = start;
		xs[b * 2 + 1] = Math.max(start, end - 1);
		for (let c = 0; c < channels.length; c++) {
			const ch = channels[c];
			let mn = Infinity;
			let mx = -Infinity;
			for (let i = start; i < end; i++) {
				const v = Number(ch[i]);
				if (v < mn) mn = v;
				if (v > mx) mx = v;
			}
			ys[c][b * 2] = mn;
			ys[c][b * 2 + 1] = mx;
		}
	}
	return { xs, ys };
}
