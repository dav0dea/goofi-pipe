import { describe, expect, it } from 'vitest';
import { PerfStats } from './perfStats.svelte';

describe('PerfStats', () => {
	it('mirrors the meter paint rate into a reactive field after a tick', () => {
		// Only the paint rate lives here. A coalesced frame belongs to the STREAM whose frame was
		// overwritten, so it is counted per (node, slot) in `frames.dropRate` — see that file's
		// tests, which use two streams precisely so per-stream and app-wide can be told apart.
		const p = new PerfStats(0);
		for (let i = 0; i < 30; i++) p.delivered();
		p.tick(1000);
		expect(p.fps).toBeCloseTo(30, 1);
	});

	it('stays at zero until a full window has elapsed', () => {
		const p = new PerfStats(0);
		p.delivered();
		p.tick(100);
		expect(p.fps).toBe(0);
	});
});
