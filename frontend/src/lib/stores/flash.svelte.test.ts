import { describe, it, expect, vi } from 'vitest';
import { Flash } from './flash.svelte';

describe('Flash', () => {
	it('marks pulsed nodes active and auto-clears them after the duration', () => {
		vi.useFakeTimers();
		try {
			const f = new Flash();
			f.pulse(['osc0', 'buf0'], 700);
			expect(f.active('osc0')).toBe(true);
			expect(f.active('buf0')).toBe(true);
			expect(f.active('other')).toBe(false);
			vi.advanceTimersByTime(699);
			expect(f.active('osc0')).toBe(true);
			vi.advanceTimersByTime(1);
			expect(f.active('osc0')).toBe(false);
			expect(f.active('buf0')).toBe(false);
		} finally {
			vi.useRealTimers();
		}
	});

	it('ignores an empty pulse', () => {
		const f = new Flash();
		f.pulse([], 700);
		expect(f.active('x')).toBe(false);
	});

	it('re-pulsing a still-active node extends its window from the new call', () => {
		vi.useFakeTimers();
		try {
			const f = new Flash();
			f.pulse(['osc0'], 700);
			vi.advanceTimersByTime(500);
			f.pulse(['osc0'], 700); // restart the clock
			vi.advanceTimersByTime(500); // 1000 since first, 500 since second
			expect(f.active('osc0')).toBe(true);
			vi.advanceTimersByTime(200); // 700 since second
			expect(f.active('osc0')).toBe(false);
		} finally {
			vi.useRealTimers();
		}
	});
});
