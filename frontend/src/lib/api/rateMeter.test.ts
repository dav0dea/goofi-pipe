import { describe, expect, it } from 'vitest';
import { RateMeter } from './rateMeter';

describe('RateMeter', () => {
	it('computes per-second rates over its window on tick', () => {
		const m = new RateMeter(0);
		for (let i = 0; i < 30; i++) m.delivered();
		for (let i = 0; i < 6; i++) m.dropped();
		m.tick(1000); // 1s window → 30 fps, 6 drops/s
		expect(m.fps).toBeCloseTo(30, 1);
		expect(m.dps).toBeCloseTo(6, 1);
	});

	it('does not recompute before the minimum window elapses', () => {
		const m = new RateMeter(0);
		m.delivered();
		m.tick(100); // < 500ms → no update yet
		expect(m.fps).toBe(0);
	});

	it('resets counters after a window so rates reflect only the latest window', () => {
		const m = new RateMeter(0);
		for (let i = 0; i < 60; i++) m.delivered();
		m.tick(1000); // 60 fps
		expect(m.fps).toBeCloseTo(60, 1);
		m.tick(2000); // no deliveries in the 2nd window → 0 fps
		expect(m.fps).toBeCloseTo(0, 1);
	});
});
