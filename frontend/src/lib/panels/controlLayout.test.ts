import { describe, it, expect } from 'vitest';
import {
	cellAt,
	freeCell,
	freshName,
	movedBy,
	overlaps,
	resizedBy,
	snap,
	turnedBy,
	type Cell
} from './controlLayout';

/* The control panel's decisions, driven the way the component drives them. The component cannot
 * mount in vitest, so everything that DECIDES lives here and everything that draws lives there. */

const square = { x: 48, y: 48 };

describe('snap', () => {
	it('rounds to the grid and never lets a widget off the top or left edge', () => {
		expect(snap({ x: 2.4, y: 1.6, w: 2.2, h: 1.9 })).toEqual({ x: 2, y: 2, w: 2, h: 2 });
		expect(snap({ x: -3, y: -0.4, w: 2, h: 2 })).toEqual({ x: 0, y: 0, w: 2, h: 2 });
	});

	it('never lets a widget collapse to nothing', () => {
		expect(snap({ x: 0, y: 0, w: 0, h: -4 })).toEqual({ x: 0, y: 0, w: 1, h: 1 });
	});
});

describe('overlaps', () => {
	const a: Cell = { x: 0, y: 0, w: 2, h: 2 };
	it('is true only where the two cells share a square', () => {
		expect(overlaps(a, { x: 1, y: 1, w: 2, h: 2 })).toBe(true);
		// Edge to edge is not an overlap: a widget may sit right beside another.
		expect(overlaps(a, { x: 2, y: 0, w: 2, h: 2 })).toBe(false);
		expect(overlaps(a, { x: 0, y: 2, w: 2, h: 2 })).toBe(false);
	});
});

describe('freeCell', () => {
	it('puts a new widget in the first free square, in reading order', () => {
		expect(freeCell([], 2, 2, 6)).toEqual({ x: 0, y: 0, w: 2, h: 2 });
		expect(freeCell([{ x: 0, y: 0, w: 2, h: 2 }], 2, 2, 6)).toEqual({ x: 2, y: 0, w: 2, h: 2 });
	});

	it('wraps to the next row when the row is full, and never lands on a taken cell', () => {
		const taken: Cell[] = [
			{ x: 0, y: 0, w: 2, h: 2 },
			{ x: 2, y: 0, w: 2, h: 2 }
		];
		const got = freeCell(taken, 2, 2, 4);
		expect(got).toEqual({ x: 0, y: 2, w: 2, h: 2 });
		expect(taken.some((t) => overlaps(t, got))).toBe(false);
	});

	it('narrows a widget too wide for the grid rather than placing it off the edge', () => {
		expect(freeCell([], 9, 1, 4)).toEqual({ x: 0, y: 0, w: 4, h: 1 });
	});
});

describe('a drop from the palette', () => {
	it('lands the widget centred under the finger', () => {
		// The centre of a 2×2 at (144, 96) is its top-left at (96, 48): cell (2, 1).
		expect(cellAt(144, 96, 2, 2, square, 8)).toEqual({ x: 2, y: 1, w: 2, h: 2 });
	});

	it('keeps a widget dropped at the edge on the board', () => {
		expect(cellAt(400, 20, 4, 1, square, 8)).toEqual({ x: 4, y: 0, w: 4, h: 1 });
		expect(cellAt(-50, -50, 2, 2, square, 8)).toEqual({ x: 0, y: 0, w: 2, h: 2 });
	});

	it('reads rows and columns as two units, because a row may be taller than a column is wide', () => {
		expect(cellAt(48, 120, 1, 1, { x: 48, y: 60 }, 8)).toEqual({ x: 1, y: 2, w: 1, h: 1 });
	});
});

describe('a widget is born with a fresh name', () => {
	it('counts from zero and skips what the group already holds', () => {
		expect(freshName('knob', [])).toBe('knob0');
		expect(freshName('knob', ['knob0', 'knob1', 'slider0'])).toBe('knob2');
		expect(freshName('slider', ['slider0', 'slider2'])).toBe('slider1');
	});
});

describe('a drag in edit mode', () => {
	const origin: Cell = { x: 1, y: 1, w: 2, h: 2 };
	it('moves the widget by whole grid units', () => {
		expect(movedBy(origin, 96, 48, square, 8)).toEqual({ x: 3, y: 2, w: 2, h: 2 });
	});

	it('stops at the right edge instead of leaving the board', () => {
		expect(movedBy(origin, 4000, 0, square, 8)).toEqual({ x: 6, y: 1, w: 2, h: 2 });
	});

	it('resizes the far edge and leaves the near one where it is', () => {
		expect(resizedBy(origin, 48, 0, square, 8)).toEqual({ x: 1, y: 1, w: 3, h: 2 });
		expect(resizedBy(origin, 4000, 0, square, 8)).toEqual({ x: 1, y: 1, w: 7, h: 2 });
	});
});

describe('a drag out of edit mode', () => {
	it('turns a value up when the drag goes up, and down when it goes down', () => {
		expect(turnedBy(0.5, -80, 0, 1, 0.01)).toBeCloseTo(1);
		expect(turnedBy(0.5, 80, 0, 1, 0.01)).toBeCloseTo(0);
	});

	it('holds the value inside its range whatever the drag', () => {
		expect(turnedBy(0.5, -4000, 0, 1, 0.01)).toBe(1);
		expect(turnedBy(0.5, 4000, 0, 1, 0.01)).toBe(0);
	});

	it('lands on the step, with no float tail', () => {
		expect(turnedBy(0, -16, 0, 1, 0.1)).toBe(0.1);
		expect(turnedBy(0, -16, 0, 10, 1)).toBe(1);
	});

	it('reads the RANGE, not the screen: a wide range moves further for the same drag', () => {
		const narrow = turnedBy(0, -16, 0, 1, 0.01);
		const wide = turnedBy(0, -16, 0, 100, 0.01);
		expect(wide).toBeGreaterThan(narrow * 10);
	});
});
