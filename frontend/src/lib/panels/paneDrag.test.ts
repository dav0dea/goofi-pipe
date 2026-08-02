import { describe, it, expect } from 'vitest';
import { PANE_AXES, coordOf, paneSizeAt, type PaneDrag } from './paneDrag';

/* The inspector pane's resize arithmetic, which is the whole of what a unit test can reach: the
   component cannot be mounted in this repo's vitest, so the rule "a drag toward the pane shrinks it,
   on whichever axis, and never below the floor" has to be provable here or nowhere. */

const drag = (over: Partial<PaneDrag> = {}): PaneDrag => ({
	startSize: 400,
	startPos: 900,
	...over
});

describe('paneDrag', () => {
	it('reads the coordinate its axis is on, and only that one', () => {
		const e = { clientX: 12, clientY: 34 };
		expect(coordOf('x', e)).toBe(12);
		expect(coordOf('y', e)).toBe(34);
	});

	/* The two anchors are one gesture, not two: the pane sits against the right edge or the bottom
	   one with its grip on the leading edge, so pushing the grip INTO the pane shrinks it either way
	   — and this module cannot tell which, because a `PaneDrag` carries no axis at all. `PANE_AXES`
	   and `coordOf` are where the axis is spent; by the time a drag exists it is two numbers and a
	   floor that mean the same thing on either anchor. */
	it('shrinks as the grip is pushed toward the pane, whichever axis picked the numbers', () => {
		expect(paneSizeAt(drag(), 960)).toBe(340);
		expect(paneSizeAt(drag({ startSize: 445, startPos: 400 }), 485)).toBe(360);
	});

	it('grows as the grip is pulled away from it', () => {
		expect(paneSizeAt(drag(), 840)).toBe(460);
	});

	it('does not move when the pointer does not', () => {
		expect(paneSizeAt(drag(), 900)).toBe(400);
	});

	/* NEITHER bound is this module's, and that is the whole of why it is three lines. Both are one
	   `clamp(10%, …, 90%)` per axis in `InspectorOverlay.svelte` — host-relative, so only the
	   stylesheet can evaluate them, and written together, so the floor cannot end up above the
	   ceiling. A floor HERE was a second, independently-authored answer, and the two contradicted
	   each other: 260px of floor under a ceiling that resolved to 256px on a landscape phone, an
	   EMPTY range. So a pointer dragged past either bound is reported UNCLAMPED; what the pane
	   actually becomes is CSS's answer, and what the gesture persists is the rendered box. */
	it('imposes neither bound — a pointer past the floor is reported unclamped', () => {
		expect(paneSizeAt(drag(), 1040)).toBe(260);
		expect(paneSizeAt(drag(), 5000)).toBe(-3700);
	});

	it('…nor a ceiling, however far the grip is pulled out', () => {
		expect(paneSizeAt(drag(), -5000)).toBe(6300);
	});
});

/* Everything the AXIS selects, in one record — which dimension of a box sizes the pane, and which
   key remembers it. The point of it being a record is that the answer is looked up once, at
   pointerdown, instead of the same question ("is this the vertical one?") being asked again at
   every line of the gesture that happens to need a dimension. Scattered that way it read as
   orientation threaded through the drag; it is one fact.

   ORIENTATION picks the record, and nothing else in this module has an opinion about it. */
describe('PANE_AXES', () => {
	it('sizes each axis by its own dimension of a box', () => {
		const box = { width: 400, height: 300 };
		expect(PANE_AXES.x.sizeOf(box)).toBe(400);
		expect(PANE_AXES.y.sizeOf(box)).toBe(300);
	});

	it('gives each axis its own key to be remembered under (D-I3)', () => {
		expect(PANE_AXES.x.key).not.toBe(PANE_AXES.y.key);
	});
});
