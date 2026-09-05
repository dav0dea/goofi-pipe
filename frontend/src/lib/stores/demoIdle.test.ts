import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { DemoIdle } from './demoIdle.svelte';

const announce = (closing_in_ms: number | null) => ({ event: 'demo_idle', payload: { closing_in_ms } }) as const;

describe('the demo idle countdown', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('opens on an announcement, counts down, and withdraws on the retraction', () => {
		const ctl = new FakeControl();
		const idle = new DemoIdle(ctl);
		expect(idle.pending).toBe(false);

		ctl.emit(announce(60_000));
		expect(idle.pending).toBe(true);
		expect(idle.secondsLeft).toBe(60);

		vi.advanceTimersByTime(30_000);
		expect(idle.secondsLeft).toBe(30);

		// A peer spoke: the manager withdraws it, and this tab's dialog closes with it.
		ctl.emit(announce(null));
		expect(idle.pending).toBe(false);
		expect(idle.secondsLeft).toBe(0);
	});

	it('never counts below zero once the deadline passes', () => {
		const ctl = new FakeControl();
		const idle = new DemoIdle(ctl);
		ctl.emit(announce(1_000));
		vi.advanceTimersByTime(5_000);
		expect(idle.secondsLeft).toBe(0);
		// Still pending: only the manager closing the socket ends it, never the client's own clock.
		expect(idle.pending).toBe(true);
	});

	it('stays by speaking through the one door, not a channel of its own', async () => {
		const ctl = new FakeControl();
		const idle = new DemoIdle(ctl);
		ctl.emit(announce(60_000));
		await idle.stay();
		expect(idle.pending).toBe(false);
		expect(ctl.recordedCalls().map((c) => c.op)).toEqual(['session status']);
	});
});
