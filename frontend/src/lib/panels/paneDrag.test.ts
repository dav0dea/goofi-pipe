import { describe, it, expect } from 'vitest';
import { coordOf, paneSizeAt, type PaneDrag } from './paneDrag';

/* The inspector pane's resize arithmetic, which is the whole of what a unit test can reach: the
   component cannot be mounted in this repo's vitest, so the rule "a drag toward the pane shrinks it,
   on whichever axis, and never below the floor" has to be provable here or nowhere. */

const drag = (over: Partial<PaneDrag> = {}): PaneDrag => ({
	axis: 'x',
	startSize: 400,
	startPos: 900,
	min: 260,
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
	   — and the arithmetic is identical once the axis has picked the coordinate. */
	it('shrinks as the grip is pushed toward the pane, on either axis', () => {
		expect(paneSizeAt(drag({ axis: 'x' }), 960)).toBe(340);
		expect(paneSizeAt(drag({ axis: 'y' }), 960)).toBe(340);
	});

	it('grows as the grip is pulled away from it', () => {
		expect(paneSizeAt(drag(), 840)).toBe(460);
	});

	it('does not move when the pointer does not', () => {
		expect(paneSizeAt(drag(), 900)).toBe(400);
	});

	/* The FLOOR is this module's; the ceiling is deliberately not. `max-width: min(30%, 30rem)` and
	   `max-height: 60%` are host- and rem-relative, so the stylesheet is the only place that can
	   evaluate them — a number here would be a second answer to one question, which is exactly what
	   `MAX_PANEL_WIDTH = 720` was, sitting above a host clamp that always bound first. */
	it('clamps at the floor and never below it, however far the grip is pushed', () => {
		expect(paneSizeAt(drag(), 1040)).toBe(260);
		expect(paneSizeAt(drag(), 5000)).toBe(260);
	});

	it('imposes no ceiling of its own — that is the stylesheet’s (D-I6)', () => {
		expect(paneSizeAt(drag(), -5000)).toBe(6300);
	});

	it('takes its own floor, so the two axes can differ', () => {
		expect(paneSizeAt(drag({ axis: 'y', startSize: 445, startPos: 400, min: 160 }), 900)).toBe(160);
	});
});
