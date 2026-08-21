import type { ArrayData } from '$lib/codec/decode';

export interface ViewSummary {
	shape: number[];
	dtype: string;
	min: number | null;
	mean: number | null;
	max: number | null;
}

/** Shape/dtype + min/mean/max for a non-renderable frame, from its float values. */
export function summaryOf(arraySpec: ArrayData): ViewSummary {
	const v = arraySpec.values;
	let mn = Infinity;
	let mx = -Infinity;
	let sum = 0;
	let n = 0;
	for (let i = 0; i < v.length; i++) {
		const x = Number(v[i]);
		if (!Number.isFinite(x)) continue;
		if (x < mn) mn = x;
		if (x > mx) mx = x;
		sum += x;
		n++;
	}
	return {
		shape: arraySpec.shape,
		dtype: arraySpec.dtype,
		min: n ? mn : null,
		mean: n ? sum / n : null,
		max: n ? mx : null
	};
}
