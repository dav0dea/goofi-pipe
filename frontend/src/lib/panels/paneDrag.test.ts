import { describe, it, expect } from 'vitest';
import {
	DISMISS_OVERSHOOT_PX,
	PANE_AXES,
	coordOf,
	endsInDismiss,
	paneSizeAt,
	type PaneDrag
} from './paneDrag';

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

/* Everything the AXIS selects, in one record — which dimension of a box sizes the pane, where its
   floor is, and which key remembers it. The point of it being a record is that the answer is looked
   up once, at pointerdown, instead of the same question ("is this the vertical one?") being asked
   again at every line of the gesture that happens to need a dimension. Scattered that way it read
   as orientation threaded through the drag; it is one fact.

   ORIENTATION picks the record. INPUT MODALITY never touches it — `endsInDismiss` is the whole of
   what modality gates, and it takes no axis knowledge at all. */
describe('PANE_AXES', () => {
	it('sizes each axis by its own dimension of a box', () => {
		const box = { width: 400, height: 300 };
		expect(PANE_AXES.x.sizeOf(box)).toBe(400);
		expect(PANE_AXES.y.sizeOf(box)).toBe(300);
	});

	it('gives each axis its own floor, and its own key to be remembered under (D-I3)', () => {
		expect(PANE_AXES.x.min).toBe(260);
		expect(PANE_AXES.y.min).toBe(160);
		expect(PANE_AXES.x.key).not.toBe(PANE_AXES.y.key);
	});

	/* The floors here are the ones the gesture actually clamps to — the same numbers, not a second
	   set that could drift from them. */
	it('is where the floor the gesture clamps to comes from', () => {
		for (const axis of ['x', 'y'] as const) {
			const { min } = PANE_AXES[axis];
			expect(paneSizeAt(drag({ axis, startSize: min + 100, min }), 5000)).toBe(min);
		}
	});
});

/* D-I4's ONE modality gate, and the reason it can be a pure function at all: the pane clamps at its
   floor, so on screen a drag that bottoms out and a drag that keeps going are the same picture. What
   tells them apart is the UNCLAMPED size the pointer asked for — which is why `paneSizeAt` is not
   the only thing this module computes. */
describe('endsInDismiss', () => {
	const sheet: PaneDrag = { axis: 'y', startSize: 445, startPos: 400, min: 160 };
	/** The coordinate at which the pointer has asked for `size`. */
	const asking = (size: number): number => sheet.startPos + (sheet.startSize - size);

	it('closes the pane when a TOUCH pulls a full overshoot past the floor', () => {
		expect(endsInDismiss(sheet, asking(sheet.min - DISMISS_OVERSHOOT_PX), 'touch')).toBe(true);
		expect(endsInDismiss(sheet, asking(-500), 'touch')).toBe(true);
	});

	/* A resize that merely bottoms out is a resize. Without the overshoot every drag to the floor
	   would close the pane, which is the opposite of a continuous resize that remembers where it was
	   left (D-I3). */
	it('does not close it for a drag that only reaches the floor', () => {
		expect(endsInDismiss(sheet, asking(sheet.min), 'touch')).toBe(false);
		expect(endsInDismiss(sheet, asking(sheet.min - DISMISS_OVERSHOOT_PX + 1), 'touch')).toBe(false);
	});

	it('never closes it for a drag that grows the pane, however far', () => {
		expect(endsInDismiss(sheet, asking(5000), 'touch')).toBe(false);
	});

	/* The gate. Everything else about this gesture is identical on both inputs — the anchor, the
	   clamp, the persistence — and the swipe is the one thing layered on top of it for touch, because
	   a mouse already has the ✕ and the escape hatch it has always had. A mouse dragged to the far
	   side of the screen must resize to the floor and stay open. */
	it('is touch-only: the same drag under a mouse or a pen resizes and stays open', () => {
		for (const pointerType of ['mouse', 'pen', ''])
			expect(endsInDismiss(sheet, asking(-500), pointerType), pointerType).toBe(false);
	});

	it('applies to the right-hand pane too — the swipe follows the anchor, not the device', () => {
		const pane: PaneDrag = { axis: 'x', startSize: 420, startPos: 900, min: 260 };
		expect(endsInDismiss(pane, 900 + 420 - (260 - DISMISS_OVERSHOOT_PX), 'touch')).toBe(true);
	});
});
