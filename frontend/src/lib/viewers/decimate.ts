/**
 * Min/max decimation for line plots: each channel folds to two points per target column.
 * `base` is the sample-index origin — 1 under log-x, since log10(0) collapses uPlot's x-scale.
 */
export interface Decimated {
	xs: number[];
	ys: number[][];
}

export function decimateMinMax(
	channels: ArrayLike<number>[],
	m: number,
	targetCols: number,
	base = 0
): Decimated {
	const buckets = Math.max(1, Math.min(targetCols, m));
	const bucketSize = m / buckets;
	const xs = new Array<number>(buckets * 2);
	const ys = channels.map(() => new Array<number>(buckets * 2));

	for (let b = 0; b < buckets; b++) {
		const start = Math.floor(b * bucketSize);
		const end = Math.min(m, Math.floor((b + 1) * bucketSize));
		xs[b * 2] = start + base;
		xs[b * 2 + 1] = Math.max(start, end - 1) + base;
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
