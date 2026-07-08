import { describe, it, expect } from 'vitest';
import { decimateMinMax } from './decimate';

describe('decimateMinMax', () => {
	it('reduces to <= 2 points per target column', () => {
		const ch = Array.from({ length: 1000 }, (_, i) => i);
		const out = decimateMinMax([ch], 1000, 100);
		expect(out.xs.length).toBeLessThanOrEqual(200);
		expect(out.ys[0].length).toBe(out.xs.length);
	});

	it('preserves per-bucket min and max (spikes survive)', () => {
		// 8 samples, 2 buckets of 4: bucket0=[0,5,1,2] -> min0,max5; bucket1=[0,9,1,2] -> min0,max9
		const out = decimateMinMax([[0, 5, 1, 2, 0, 9, 1, 2]], 8, 2);
		expect(out.ys[0]).toEqual([0, 5, 0, 9]);
	});

	it('keeps x strictly increasing (uPlot requires it)', () => {
		const ch = Array.from({ length: 777 }, (_, i) => Math.sin(i));
		const out = decimateMinMax([ch], 777, 80);
		for (let i = 1; i < out.xs.length; i++) {
			expect(out.xs[i]).toBeGreaterThan(out.xs[i - 1]);
		}
	});

	it('offsets every x by base so log-x has no x <= 0', () => {
		// Under Log X the viewer sets uPlot x.distr=3 (log10); log10(0) is -Infinity,
		// which collapses the whole x-scale and blanks the plot. Bucket 0 starts at
		// sample 0, so without a base decimation emits xs[0]=0 — the log-x empty-plot
		// bug. `base` mirrors indexAxis's 1-based rule so the decimated path stays >0.
		const ch = Array.from({ length: 1000 }, (_, i) => i);
		const out = decimateMinMax([ch], 1000, 100, 1);
		expect(out.xs[0]).toBe(1);
		for (const x of out.xs) expect(x).toBeGreaterThan(0);
	});

	it('defaults base to 0 (linear x unchanged)', () => {
		const out = decimateMinMax([[0, 5, 1, 2, 0, 9, 1, 2]], 8, 2);
		expect(out.xs[0]).toBe(0);
	});

	it('handles multiple channels with a shared x grid', () => {
		const a = [0, 1, 2, 3];
		const b = [9, 8, 7, 6];
		const out = decimateMinMax([a, b], 4, 2);
		expect(out.ys).toHaveLength(2);
		expect(out.ys[0].length).toBe(out.xs.length);
		expect(out.ys[1].length).toBe(out.xs.length);
		// bucket0 of a=[0,1]->min0,max1; bucket1=[2,3]->min2,max3
		expect(out.ys[0]).toEqual([0, 1, 2, 3]);
		// bucket0 of b=[9,8]->min8,max9; bucket1=[7,6]->min6,max7
		expect(out.ys[1]).toEqual([8, 9, 6, 7]);
	});

	it('coerces bigint-typed values to number', () => {
		const out = decimateMinMax([[0n, 5n, 1n, 9n] as unknown as ArrayLike<number>], 4, 2);
		expect(out.ys[0]).toEqual([0, 5, 1, 9]);
	});
});
