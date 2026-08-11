import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DemandTicker } from './demandTicker';

/**
 * The data worker's decode ticker was a module-level `setInterval` started once at load and never
 * cleared, so it woke the worker ~62x/s for the rest of the page's life after the last viewer
 * closed. `dataWorker.ts` cannot be imported here (it touches `self` at module scope, which the
 * node test environment does not provide), so the lifecycle it got wrong lives here instead —
 * where it can actually be driven.
 */
describe('DemandTicker', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('does not run before there is any demand', () => {
		const fn = vi.fn();
		const t = new DemandTicker(fn, 16);
		expect(t.armed).toBe(false);
		vi.advanceTimersByTime(1000);
		expect(fn).not.toHaveBeenCalled();
	});

	it('arms on the first demand and keeps running while demand holds', () => {
		const fn = vi.fn();
		const t = new DemandTicker(fn, 16);
		t.sync(1);
		expect(t.armed).toBe(true);
		vi.advanceTimersByTime(64);
		expect(fn).toHaveBeenCalledTimes(4);
		// More demand is not more timers — the ticker drains everything in one pass.
		t.sync(3);
		vi.advanceTimersByTime(16);
		expect(fn).toHaveBeenCalledTimes(5);
		expect(vi.getTimerCount()).toBe(1);
	});

	it('clears with the last demand and stops firing entirely', () => {
		const fn = vi.fn();
		const t = new DemandTicker(fn, 16);
		t.sync(1);
		vi.advanceTimersByTime(32);
		const before = fn.mock.calls.length;

		t.sync(0);
		expect(t.armed).toBe(false);
		expect(vi.getTimerCount(), 'no timer is left pending').toBe(0);
		vi.advanceTimersByTime(10_000);
		expect(fn.mock.calls.length, 'nothing fires once demand is gone').toBe(before);
	});

	it('re-arms when demand returns', () => {
		const fn = vi.fn();
		const t = new DemandTicker(fn, 16);
		t.sync(1);
		t.sync(0);
		t.sync(1);
		expect(t.armed).toBe(true);
		vi.advanceTimersByTime(16);
		expect(fn).toHaveBeenCalledTimes(1);
		expect(vi.getTimerCount(), 'and does not stack a second interval').toBe(1);
	});
});
