import { describe, expect, it } from 'vitest';
import type { ArrayData } from '$lib/codec/decode';
import { summaryOf } from './viewMeta';

const arr = (dtype: string, shape: number[], values: number[]): ArrayData => ({
	dtype,
	shape,
	values
});

describe('summaryOf', () => {
	it('computes shape/dtype/min/mean/max from the (float-accurate) array', () => {
		const s = summaryOf(arr('<f4', [3], [1, 2, 3]));
		expect(s.shape).toEqual([3]);
		expect(s.dtype).toBe('<f4');
		expect(s.min).toBe(1);
		expect(s.max).toBe(3);
		expect(s.mean).toBe(2);
	});

	it('ignores non-finite values and reports null stats for an empty array', () => {
		expect(summaryOf(arr('<f4', [2], [NaN, Infinity])).min).toBeNull();
		const s = summaryOf(arr('<f4', [0], []));
		expect(s.min).toBeNull();
		expect(s.mean).toBeNull();
		expect(s.max).toBeNull();
	});
});
