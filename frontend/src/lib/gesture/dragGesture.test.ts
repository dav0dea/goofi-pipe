import { describe, it, expect } from 'vitest';
import { beginDrag, type DragPointerTarget } from './dragGesture';

/** A DOM-shaped stub: the module only ever captures the pointer and (un)binds three events. */
function target() {
	const bound = new Map<string, Set<(e: PointerEvent) => void>>();
	let captured: number | null = null;
	const el: DragPointerTarget = {
		setPointerCapture: (id) => (captured = id),
		addEventListener: (t, fn) => void (bound.get(t) ?? bound.set(t, new Set()).get(t)!).add(fn),
		removeEventListener: (t, fn) => void bound.get(t)?.delete(fn)
	};
	return {
		el,
		get captured() {
			return captured;
		},
		/** How many listeners are still attached, across all three types. */
		attached: () => [...bound.values()].reduce((n, s) => n + s.size, 0),
		fire: (t: string, e = {} as PointerEvent) => [...(bound.get(t) ?? [])].forEach((fn) => fn(e))
	};
}

function spies() {
	const calls: string[] = [];
	return {
		calls,
		handlers: {
			move: () => calls.push('move'),
			commit: () => calls.push('commit'),
			cancel: () => calls.push('cancel')
		}
	};
}

describe('beginDrag', () => {
	it('captures the pointer and binds move / up / cancel', () => {
		const t = target();
		beginDrag(t.el, 7, spies().handlers);
		expect(t.captured).toBe(7);
		expect(t.attached()).toBe(3);
	});

	it('feeds moves through, then commits on release and detaches', () => {
		const t = target();
		const s = spies();
		beginDrag(t.el, 1, s.handlers);
		t.fire('pointermove');
		t.fire('pointerup');
		expect(s.calls).toEqual(['move', 'commit']);
		expect(t.attached(), 'a resolved gesture leaves nothing bound').toBe(0);
	});

	it('CANCELS — never commits — on pointercancel, and detaches', () => {
		// The whole defect this exists for: a touch pointer is cancelled, not released, whenever the
		// browser reclaims the gesture. Handling only `pointerup` left the listeners live, so the next
		// release anywhere in the app committed the abandoned intent (a panel split, or a close).
		const t = target();
		const s = spies();
		beginDrag(t.el, 1, s.handlers);
		t.fire('pointermove');
		t.fire('pointercancel');
		expect(s.calls).toEqual(['move', 'cancel']);
		expect(t.attached()).toBe(0);

		// …and the release that follows the cancelled gesture reaches nothing.
		t.fire('pointerup');
		expect(s.calls).toEqual(['move', 'cancel']);
	});

	it('returns a teardown that cancels an in-flight gesture (the unmount path)', () => {
		const t = target();
		const s = spies();
		const teardown = beginDrag(t.el, 1, s.handlers);
		teardown();
		expect(s.calls).toEqual(['cancel']);
		expect(t.attached()).toBe(0);
	});

	it('is idempotent once resolved — a teardown after the fact commits nothing more', () => {
		const t = target();
		const s = spies();
		const teardown = beginDrag(t.el, 1, s.handlers);
		t.fire('pointerup');
		teardown();
		expect(s.calls, 'the unmount teardown of a finished gesture is a no-op').toEqual(['commit']);
	});
});
