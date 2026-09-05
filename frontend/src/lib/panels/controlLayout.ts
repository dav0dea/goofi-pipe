/** The control panel's geometry, kept out of the component so a test can drive it. A cell is in
 * GRID units, never pixels: the panel's width decides what a unit is worth. The drag LAW that turns
 * a widget lives with the widget, in `$lib/ui/knob`. */

export { turnedBy } from '$lib/ui/knob';

export interface Cell {
	x: number;
	y: number;
	w: number;
	h: number;
}

/** How many columns the board is, whatever its pixel width. */
export const COLUMNS = 8;

/** The smallest a widget may be, in grid units. */
export const MIN_W = 1;
export const MIN_H = 1;

/** A cell on the grid, never narrower than the minimum and never off the left or top edge. */
export function snap(cell: Cell): Cell {
	return {
		x: Math.max(0, Math.round(cell.x)),
		y: Math.max(0, Math.round(cell.y)),
		w: Math.max(MIN_W, Math.round(cell.w)),
		h: Math.max(MIN_H, Math.round(cell.h))
	};
}

export function overlaps(a: Cell, b: Cell): boolean {
	return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/** Where a new widget lands: the first free cell in reading order, on a grid `columns` wide. */
export function freeCell(taken: Cell[], w: number, h: number, columns: number): Cell {
	const width = Math.max(MIN_W, Math.min(w, columns));
	for (let y = 0; ; y++) {
		for (let x = 0; x + width <= columns; x++) {
			const want = { x, y, w: width, h: Math.max(MIN_H, h) };
			if (!taken.some((t) => overlaps(t, want))) return want;
		}
	}
}

/** The cell a drag from `origin` by `dx, dy` pixels lands on, with `unit` pixels to the grid unit. */
export function movedBy(origin: Cell, dx: number, dy: number, unit: number): Cell {
	return snap({ ...origin, x: origin.x + dx / unit, y: origin.y + dy / unit });
}

/** The cell a resize drag lands on. The origin's x and y stay: a resize moves the far edge only. */
export function resizedBy(origin: Cell, dx: number, dy: number, unit: number): Cell {
	return snap({ ...origin, w: origin.w + dx / unit, h: origin.h + dy / unit });
}
