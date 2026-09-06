/** The control panel's geometry, kept out of the component so a test can drive it. A cell is in
 * GRID units, never pixels: the panel's width decides what a unit is worth. The drag LAW that turns
 * a widget lives with the widget, in `$lib/ui/knob`. */

import type { GlobalType } from '$lib/crdt/graphDoc';
import { CONTROL_COLUMNS, CONTROL_KINDS, type ControlKindId } from '$lib/api/vocab';

export { turnedBy } from '$lib/ui/knob';

export type Kind = ControlKindId;
export const KINDS: Kind[] = CONTROL_KINDS.map((k) => k.id);

/** The value type each widget draws, so a widget asks for one thing, not two. */
export const TYPE_OF: Record<Kind, GlobalType> = Object.fromEntries(CONTROL_KINDS.map((k) => [k.id, k.type])) as Record<Kind, GlobalType>;

/** The box a widget is born in, in grid units. */
export const BORN: Record<Kind, { w: number; h: number }> = Object.fromEntries(
	CONTROL_KINDS.map((k) => [k.id, { w: k.w, h: k.h }])
) as Record<Kind, { w: number; h: number }>;

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
export const COLUMNS = CONTROL_COLUMNS;

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

export function sameCell(a: Cell, b: Cell): boolean {
	return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}

export function overlaps(a: Cell, b: Cell): boolean {
	return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/** The cell a `w × h` widget lands on when dropped with its centre `px, py` pixels into the
 * board. It may cover another — a drop is free — but never hangs off the right edge. */
export function cellAt(px: number, py: number, w: number, h: number, units: Units, columns: number): Cell {
	return inside(snap({ x: px / units.x - w / 2, y: py / units.y - h / 2, w, h }), columns);
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
