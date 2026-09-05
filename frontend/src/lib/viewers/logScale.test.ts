import { describe, it, expect } from 'vitest';
import { logSafe } from './logScale';

/** uPlot's log tick generator walks `l += incr` from the minimum until it passes the maximum. A
 * non-positive minimum makes the increment -Infinity or NaN, so `l` never advances and the walk
 * never ends. This is that walk, bounded: it must terminate on whatever `logSafe` returns. */
function ticksTerminate([min, max]: [number, number]): number {
	const incr = 10 ** Math.floor(Math.log10(min));
	let n = 0;
	for (let l = min; l <= max; l += incr) {
		if (++n > 100_000) return -1;
	}
	return n;
}

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

	it('gives uPlot a range its tick walk can finish', () => {
		// The unguarded call is the defect: a zero minimum never terminates.
		expect(ticksTerminate([0, 16690.5])).toBe(-1);
		expect(ticksTerminate(logSafe(0, 16690.5))).toBeGreaterThan(0);
	});
});
