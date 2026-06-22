import { describe, it, expect, vi } from 'vitest';
import { awaitEvent } from './awaitEvent';

function bus<T>() {
	const cbs = new Set<(ev: T) => void>();
	return {
		subscribe: (cb: (ev: T) => void) => {
			cbs.add(cb);
			return () => cbs.delete(cb);
		},
		emit: (ev: T) => cbs.forEach((cb) => cb(ev)),
		size: () => cbs.size
	};
}

describe('awaitEvent', () => {
	it('resolves with the first event matching the predicate, then unsubscribes', async () => {
		const b = bus<number>();
		const p = awaitEvent(b.subscribe, (n) => n === 3, 1000);
		b.emit(1);
		b.emit(3);
		b.emit(5);
		await expect(p).resolves.toBe(3);
		expect(b.size()).toBe(0);
	});

	it('rejects on timeout and unsubscribes', async () => {
		vi.useFakeTimers();
		try {
			const b = bus<number>();
			const p = awaitEvent(b.subscribe, () => false, 500);
			const assertion = expect(p).rejects.toThrow(/timed out/i);
			await vi.advanceTimersByTimeAsync(500);
			await assertion;
			expect(b.size()).toBe(0);
		} finally {
			vi.useRealTimers();
		}
	});

	it('does not leak the subscription when an event matches synchronously on subscribe', async () => {
		// A bus that delivers a matching event during subscribe() itself.
		const cbs = new Set<(n: number) => void>();
		const subscribe = (cb: (n: number) => void): (() => void) => {
			cbs.add(cb);
			cb(7); // synchronous match
			return () => cbs.delete(cb);
		};
		await expect(awaitEvent(subscribe, (n) => n === 7, 1000)).resolves.toBe(7);
		expect(cbs.size).toBe(0);
	});
});
