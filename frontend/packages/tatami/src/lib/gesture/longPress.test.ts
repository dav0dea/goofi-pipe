import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createLongPress, HOLD_MS, MOVE_TOLERANCE_PX } from './longPress';

describe('createLongPress', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	function fired() {
		const calls: Array<{ clientX: number; clientY: number }> = [];
		return { calls, fn: (p: { clientX: number; clientY: number }) => calls.push(p) };
	}

	it('fires once the press has been held still for the hold time', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 100, clientY: 200 });
		vi.advanceTimersByTime(HOLD_MS - 1);
		expect(f.calls, 'not yet — the hold is not up').toEqual([]);
		vi.advanceTimersByTime(1);
		expect(f.calls).toEqual([{ clientX: 100, clientY: 200 }]);
	});

	it('does not fire again without a new press', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 10, clientY: 10 });
		vi.advanceTimersByTime(HOLD_MS * 4);
		expect(f.calls).toHaveLength(1);
	});

	it('a release before the hold is up disarms it', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 10, clientY: 10 });
		vi.advanceTimersByTime(HOLD_MS - 10);
		lp.cancel();
		vi.advanceTimersByTime(HOLD_MS);
		expect(f.calls).toEqual([]);
	});

	// This is what keeps the gesture out of the way of a pan: a drag moves, a press does not.
	it('a move past the tolerance disarms it — a pan is not a press', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 0, clientY: 0 });
		lp.move({ clientX: MOVE_TOLERANCE_PX + 1, clientY: 0 });
		vi.advanceTimersByTime(HOLD_MS * 2);
		expect(f.calls).toEqual([]);
	});

	it('tolerates the jitter of a finger resting on glass', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 50, clientY: 50 });
		lp.move({ clientX: 53, clientY: 47 });
		vi.advanceTimersByTime(HOLD_MS);
		// It fires at the ORIGIN, not at the drifted point: the menu belongs where the finger landed.
		expect(f.calls).toEqual([{ clientX: 50, clientY: 50 }]);
	});

	it('measures the tolerance from the origin, not from the previous move', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 0, clientY: 0 });
		// Each step is inside the tolerance, but the walk leaves it — a slow drag is still a drag.
		for (let x = 4; x <= MOVE_TOLERANCE_PX + 4; x += 4) lp.move({ clientX: x, clientY: 0 });
		vi.advanceTimersByTime(HOLD_MS * 2);
		expect(f.calls).toEqual([]);
	});

	it('a fresh press re-arms after a cancelled one', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 0, clientY: 0 });
		lp.cancel();
		lp.start({ clientX: 7, clientY: 9 });
		vi.advanceTimersByTime(HOLD_MS);
		expect(f.calls).toEqual([{ clientX: 7, clientY: 9 }]);
	});

	it('a second press replaces the first rather than stacking timers', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 0, clientY: 0 });
		vi.advanceTimersByTime(HOLD_MS - 5);
		lp.start({ clientX: 30, clientY: 30 });
		vi.advanceTimersByTime(HOLD_MS);
		expect(f.calls).toEqual([{ clientX: 30, clientY: 30 }]);
	});

	it('a move after it has fired is inert', () => {
		const f = fired();
		const lp = createLongPress(f.fn);
		lp.start({ clientX: 0, clientY: 0 });
		vi.advanceTimersByTime(HOLD_MS);
		lp.move({ clientX: 400, clientY: 400 });
		lp.cancel();
		expect(f.calls).toHaveLength(1);
	});
});
