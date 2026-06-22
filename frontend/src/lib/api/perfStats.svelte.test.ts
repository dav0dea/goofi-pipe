import { describe, expect, it } from 'vitest';
import { PerfStats } from './perfStats.svelte';

describe('PerfStats', () => {
	it('mirrors the meter rates into reactive fields after a tick', () => {
		const p = new PerfStats(0);
		for (let i = 0; i < 30; i++) p.delivered();
		for (let i = 0; i < 3; i++) p.dropped();
		p.tick(1000);
		expect(p.fps).toBeCloseTo(30, 1);
		expect(p.dps).toBeCloseTo(3, 1);
	});

	it('stays at zero until a full window has elapsed', () => {
		const p = new PerfStats(0);
		p.delivered();
		p.tick(100);
		expect(p.fps).toBe(0);
	});
});
