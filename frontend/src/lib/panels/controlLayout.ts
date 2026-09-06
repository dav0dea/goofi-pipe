/** The control panel's geometry, kept out of the component so a test can drive it. A cell is in
 * GRID units, never pixels: the panel's width decides what a unit is worth. The drag LAW that turns
 * a widget lives with the widget, in `$lib/ui/knob`. */

import type { GlobalType } from '$lib/crdt/graphDoc';

export { turnedBy } from '$lib/ui/knob';

export type Kind = 'knob' | 'slider' | 'number' | 'field' | 'toggle' | 'dropdown';
export const KINDS: Kind[] = ['knob', 'slider', 'number', 'field', 'toggle', 'dropdown'];

/** The value type each widget draws, so a widget asks for one thing, not two. */
export const TYPE_OF: Record<Kind, GlobalType> = {
	knob: 'float',
	slider: 'float',
	number: 'float',
	field: 'string',
	toggle: 'bool',
	dropdown: 'string'
};

/** The box a widget is born in, in grid units. */
export const BORN: Record<Kind, { w: number; h: number }> = {
	knob: { w: 2, h: 2 },
	slider: { w: 4, h: 1 },
	number: { w: 2, h: 1 },
	field: { w: 3, h: 1 },
	toggle: { w: 1, h: 1 },
	dropdown: { w: 3, h: 1 }
};

export interface Cell {
	x: number;
	y: number;
	w: number;
	h: number;
}

/** Pixels to one grid unit, per axis: a row is never shorter than a tap target, so it can be
 * taller than a column is wide. */
export interface Units {
	x: number;
	y: number;
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

/** `cell` pulled back inside `columns`: the width first, then the left edge. */
function inside(cell: Cell, columns: number): Cell {
	const w = Math.min(cell.w, columns);
	return { ...cell, w, x: Math.min(cell.x, columns - w) };
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

/** The cell a `w × h` widget lands on when dropped with its centre `px, py` pixels into the
 * board. It may cover another — a drop is free — but never hangs off the right edge. */
export function cellAt(px: number, py: number, w: number, h: number, units: Units, columns: number): Cell {
	return inside(snap({ x: px / units.x - w / 2, y: py / units.y - h / 2, w, h }), columns);
}

/** A fresh element name for `kind` among `taken`: the lowest `kind0`, `kind1`, … not yet used. */
export function freshName(kind: Kind, taken: string[]): string {
	for (let n = 0; ; n++) {
		const name = `${kind}${n}`;
		if (!taken.includes(name)) return name;
	}
}

/** The cell a drag from `origin` by `dx, dy` pixels lands on. */
export function movedBy(origin: Cell, dx: number, dy: number, units: Units, columns: number): Cell {
	return inside(snap({ ...origin, x: origin.x + dx / units.x, y: origin.y + dy / units.y }), columns);
}

/** The cell a resize drag lands on. The origin's x and y stay: a resize moves the far edge only. */
export function resizedBy(origin: Cell, dx: number, dy: number, units: Units, columns: number): Cell {
	const cell = snap({ ...origin, w: origin.w + dx / units.x, h: origin.h + dy / units.y });
	return { ...cell, w: Math.min(cell.w, columns - cell.x) };
}
