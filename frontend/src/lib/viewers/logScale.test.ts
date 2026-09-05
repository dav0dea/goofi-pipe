import { describe, it, expect } from 'vitest';
import { logSafe, logSplits } from './logScale';

describe('logSafe', () => {
	it('floors a zero minimum, which is what a PSD bin reaches', () => {
		const [lo, hi] = logSafe(0, 16690.5);
		expect(lo).toBeGreaterThan(0);
		expect(hi).toBe(16690.5);
	});

	it('keeps a range that is already positive', () => {
		expect(logSafe(0.5, 100)).toEqual([0.5, 100]);
	});

	it('floors a negative minimum, as a manual y-range may carry', () => {
		expect(logSafe(-200, 200)[0]).toBeGreaterThan(0);
	});

	it('answers a positive window for every degenerate input', () => {
		for (const [min, max] of [
			[0, 0],
			[-5, -1],
			[NaN, NaN],
			[-Infinity, Infinity],
			[5, 5],
			[100, 1]
		] as [number, number][]) {
			const [lo, hi] = logSafe(min, max);
			expect(lo, `lo for ${min}..${max}`).toBeGreaterThan(0);
			expect(hi, `hi for ${min}..${max}`).toBeGreaterThan(lo);
		}
	});
});

describe('logSplits', () => {
	it('places one grid line per decade across the window', () => {
		expect(logSplits(0.5, 100)).toEqual([0.1, 1, 10, 100]);
	});

	it('is bounded and ascending on every minimum that walks uPlot off its increment table', () => {
		// uPlot 1.6.32's own log walk never terminates on these: the top of a decade, and below 1e-22.
		for (const [min, max] of [
			[0.0096, 1e4],
			[9.6e-17, 1e4],
			[1.35e-30, 4.5e-16],
			[0, 16690.5],
			[NaN, NaN],
			[1e-40, 3.4e38]
		]) {
			const s = logSplits(min, max);
			expect(s.length, `count for ${min}..${max}`).toBeLessThan(100);
			expect(s.every((v, i) => i === 0 || v > s[i - 1]), `ascending for ${min}..${max}`).toBe(true);
		}
	});
});
