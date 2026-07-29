import { describe, it, expect } from 'vitest';
import { eventPoint } from './eventPoint';

describe('eventPoint', () => {
	it('reads a mouse event straight off the event', () => {
		expect(eventPoint({ clientX: 12, clientY: 34 })).toEqual({ clientX: 12, clientY: 34 });
	});

	it('reads the first live touch of a touch event', () => {
		// The bug this exists for: SvelteFlow types its drag callbacks `MouseEvent | TouchEvent`, and
		// a TouchEvent has NO `clientX` — the coordinates live one level down, per finger. The old
		// `typeof clientX !== 'number' → null` meant `linkNodeToPanel` could never fire under touch,
		// and dragging a node in is the ONLY way to populate a Viewer/Parameters/Metadata/Console panel.
		expect(eventPoint({ touches: [{ clientX: 5, clientY: 6 }], changedTouches: [] })).toEqual({
			clientX: 5,
			clientY: 6
		});
	});

	it('falls back to changedTouches, which is where a touchend puts the lifted finger', () => {
		// `touches` is EMPTY on touchend — and touchend is exactly the event that ENDS a drag, i.e.
		// the one that decides whether the node was dropped on a panel.
		expect(eventPoint({ touches: [], changedTouches: [{ clientX: 7, clientY: 8 }] })).toEqual({
			clientX: 7,
			clientY: 8
		});
	});

	it('prefers a live touch over a changed one when both are present', () => {
		expect(
			eventPoint({ touches: [{ clientX: 1, clientY: 2 }], changedTouches: [{ clientX: 9, clientY: 9 }] })
		).toEqual({ clientX: 1, clientY: 2 });
	});

	it('is null when the event carries no point at all', () => {
		expect(eventPoint({ touches: [], changedTouches: [] })).toBeNull();
		expect(eventPoint({})).toBeNull();
	});

	it('rejects a half-formed point rather than reading NaN off it', () => {
		expect(eventPoint({ clientX: 3 })).toBeNull();
	});
});
