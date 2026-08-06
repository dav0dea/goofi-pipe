import { describe, it, expect } from 'vitest';
import { createTouchPlacement, ghostOrigin, type PlacementPointer } from './touchPlacement';

const at = (
	x: number,
	y: number,
	over: Partial<PlacementPointer> = {}
): PlacementPointer => ({ pointerId: 1, pointerType: 'touch', clientX: x, clientY: y, ...over });

describe('createTouchPlacement', () => {
	it('leaves a MOUSE pointerdown alone — the desktop path is the reference and is untouched', () => {
		const g = createTouchPlacement();
		expect(g.down(at(10, 20, { pointerType: 'mouse' }), true)).toBeNull();
		expect(g.active).toBe(false);
	});

	it('leaves a PEN alone too: the gate is `touch`, not "anything that is not a mouse"', () => {
		const g = createTouchPlacement();
		expect(g.down(at(10, 20, { pointerType: 'pen' }), true)).toBeNull();
		expect(g.active).toBe(false);
	});

	it('leaves a touch OUTSIDE the canvas alone, so its synthetic click can cancel as a mouse does', () => {
		// The cancel path is not reimplemented for touch: a tap outside is simply not taken here, and
		// the compat `click` it produces reaches the same window handler a mouse click does.
		const g = createTouchPlacement();
		expect(g.down(at(10, 20), false)).toBeNull();
		expect(g.active).toBe(false);
	});

	it('takes a touch inside the canvas and puts the ghost at the finger', () => {
		const g = createTouchPlacement();
		expect(g.down(at(120, 340), true)).toEqual({ x: 120, y: 340 });
		expect(g.active).toBe(true);
	});

	it('tracks the finger while it moves', () => {
		const g = createTouchPlacement();
		g.down(at(100, 100), true);
		expect(g.move(at(140, 130))).toEqual({ x: 140, y: 130 });
		expect(g.move(at(180, 160))).toEqual({ x: 180, y: 160 });
		expect(g.active).toBe(true);
	});

	it('commits where the finger was released, and the gesture is then over', () => {
		const g = createTouchPlacement();
		g.down(at(100, 100), true);
		g.move(at(180, 160));
		expect(g.up(at(182, 161))).toEqual({ x: 182, y: 161 });
		expect(g.active).toBe(false);
	});

	it('places a plain TAP at the tap — a tap is a drag of zero length, on one path', () => {
		// The whole reason there is no tap-vs-drag threshold: nothing can fall on the wrong side of
		// a threshold that does not exist. `up` with no `move` between answers the `down` point.
		const g = createTouchPlacement();
		g.down(at(77, 88), true);
		expect(g.up(at(77, 88))).toEqual({ x: 77, y: 88 });
		expect(g.active).toBe(false);
	});

	it('ignores a SECOND finger rather than letting it teleport the ghost', () => {
		const g = createTouchPlacement();
		g.down(at(100, 100), true);
		expect(g.down(at(300, 300, { pointerId: 2 }), true)).toBeNull();
		expect(g.move(at(300, 300, { pointerId: 2 }))).toBeNull();
		expect(g.up(at(300, 300, { pointerId: 2 }))).toBeNull();
		// …and the first finger still owns it.
		expect(g.active).toBe(true);
		expect(g.move(at(150, 150))).toEqual({ x: 150, y: 150 });
	});

	it('answers nothing to a move or an up it never saw the down for', () => {
		const g = createTouchPlacement();
		expect(g.move(at(10, 10))).toBeNull();
		expect(g.up(at(10, 10))).toBeNull();
	});

	it('ends the gesture on a cancel WITHOUT committing, leaving the placement pending', () => {
		const g = createTouchPlacement();
		g.down(at(100, 100), true);
		g.cancel(at(100, 100));
		expect(g.active).toBe(false);
		expect(g.move(at(120, 120)), 'the cancelled finger no longer drives the ghost').toBeNull();
	});

	it('ignores a cancel belonging to some other pointer', () => {
		const g = createTouchPlacement();
		g.down(at(100, 100), true);
		g.cancel(at(0, 0, { pointerId: 2 }));
		expect(g.active).toBe(true);
	});

	it('is reusable: a second gesture starts cleanly after the first commits', () => {
		const g = createTouchPlacement();
		g.down(at(10, 10), true);
		g.up(at(10, 10));
		expect(g.down(at(50, 60), true)).toEqual({ x: 50, y: 60 });
		expect(g.active).toBe(true);
	});
});

describe('ghostOrigin', () => {
	// A typical ghost, in the flow units both the position and the measured size are in.
	const size = { w: 233, h: 168 };

	it('hangs a MOUSE ghost off its top-left, at the cursor — the reference, byte-for-byte', () => {
		expect(ghostOrigin({ x: 100, y: 200 }, size, 'top-left')).toEqual({ x: 100, y: 200 });
	});

	it('carries a TOUCH ghost by its middle, so the finger is over the node and not its corner', () => {
		expect(ghostOrigin({ x: 100, y: 200 }, size, 'centre')).toEqual({ x: 100 - 116.5, y: 200 - 84 });
	});

	it('centres on the MEASURED size, so a node of any height is still carried by its middle', () => {
		expect(ghostOrigin({ x: 0, y: 0 }, { w: 200, h: 400 }, 'centre')).toEqual({ x: -100, y: -200 });
	});
});
