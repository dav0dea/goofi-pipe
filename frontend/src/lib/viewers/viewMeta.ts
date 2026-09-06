import type { ArrayData } from '$lib/codec/decode';
import { isU8, sampleRange, toUnit } from './depth';

export interface ViewSummary {
	shape: number[];
	dtype: string;
	min: number | null;
	mean: number | null;
	max: number | null;
}

/** Shape/dtype + min/mean/max for a non-renderable frame, in the NODE's own units: a frame that
 * came over the 8-bit hop carries the range its texels span, and a texel is not a value. */
export function summaryOf(arraySpec: ArrayData, meta?: Record<string, unknown>): ViewSummary {
	const v = arraySpec.values;
	const range = isU8(arraySpec.dtype) ? sampleRange(meta) : null;
	let mn = Infinity;
	let mx = -Infinity;
	let sum = 0;
	let n = 0;
	for (let i = 0; i < v.length; i++) {
		const x = range ? toUnit(Number(v[i]), range) : Number(v[i]);
		if (!Number.isFinite(x)) continue;
		if (x < mn) mn = x;
		if (x > mx) mx = x;
		sum += x;
		n++;
	}
	return {
		shape: arraySpec.shape,
		// Mapped back, so what is summarized is the node's own f32, not the hop's texels.
		dtype: range ? 'float32' : arraySpec.dtype,
		min: n ? mn : null,
		mean: n ? sum / n : null,
		max: n ? mx : null
	};
}
