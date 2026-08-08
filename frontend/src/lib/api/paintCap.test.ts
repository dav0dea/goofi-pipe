import { describe, it, expect } from 'vitest';
import { MAX_VIEWER_FPS, MIN_PAINT_INTERVAL_MS, paintDelay } from './paintCap';

describe('the viewer paint cap', () => {
	it('is 30 fps, as one retunable constant', () => {
		expect(MAX_VIEWER_FPS).toBe(30);
		expect(MIN_PAINT_INTERVAL_MS).toBeCloseTo(1000 / 30, 5);
	});

	it('a first-ever flush waits nothing', () => {
		expect(paintDelay(-Infinity, 1000)).toBe(0);
	});

	it('a flush inside the cooldown waits out the remainder', () => {
		expect(paintDelay(0, 10)).toBeCloseTo(MIN_PAINT_INTERVAL_MS - 10, 5);
	});

	it('a flush at or past the interval paints immediately', () => {
		expect(paintDelay(0, MIN_PAINT_INTERVAL_MS)).toBe(0);
		expect(paintDelay(0, MIN_PAINT_INTERVAL_MS * 5)).toBe(0);
	});

	it('never returns a negative wait', () => {
		expect(paintDelay(0, 10_000)).toBe(0);
	});
});
