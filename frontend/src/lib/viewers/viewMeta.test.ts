import { describe, expect, it } from 'vitest';
import type { ArrayData } from '$lib/codec/decode';
import { manualGrayDomain, summaryOf, viewInfo } from './viewMeta';

const arr = (dtype: string, shape: number[], values: number[]): ArrayData => ({
	dtype,
	shape,
	values
});

describe('viewInfo', () => {
	it('reads meta.__view__ when present', () => {
		const v = viewInfo({ __view__: { range: [0, 1], stats: { min: 0, mean: 0.5, max: 1 } } });
		expect(v.range).toEqual([0, 1]);
		expect(v.stats?.mean).toBe(0.5);
	});

	it('returns empty info when __view__ is absent', () => {
		expect(viewInfo({ shape: [3] })).toEqual({});
		expect(viewInfo({})).toEqual({});
	});
});

describe('summaryOf', () => {
	it('prefers the backend float summary (no array body needed)', () => {
		const v = viewInfo({
			__view__: { summary: { shape: [2, 2, 2, 2], dtype: '<f4', min: 0, mean: 7.5, max: 15 } }
		});
		const s = summaryOf(arr('<f4', [0], []), v);
		expect(s.shape).toEqual([2, 2, 2, 2]);
		expect(s.dtype).toBe('<f4');
		expect(s.max).toBe(15);
	});

	it('falls back to computing stats from the array when no summary', () => {
		const s = summaryOf(arr('<f4', [3], [1, 2, 3]), {});
		expect(s.shape).toEqual([3]);
		expect(s.min).toBe(1);
		expect(s.max).toBe(3);
		expect(s.mean).toBe(2);
	});
});

describe('manualGrayDomain', () => {
	it('maps a float manual window into the uint8 data domain via __view__.range', () => {
		// data was normalized float[0,4] -> uint8[0,255]; a manual window of [1,3]
		// in float units maps to [63.75, 191.25] in the uint8 domain.
		const v = viewInfo({ __view__: { range: [0, 4] } });
		const [lo, hi] = manualGrayDomain(v, 1, 3, '|u1');
		expect(lo).toBeCloseTo(63.75, 2);
		expect(hi).toBeCloseTo(191.25, 2);
	});

	it('passes the window through unchanged for float data', () => {
		expect(manualGrayDomain({}, 0.2, 0.8, '<f4')).toEqual([0.2, 0.8]);
	});
});
