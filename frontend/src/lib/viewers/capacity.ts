/**
 * Per-viewer ViewSpec — a compatibility predicate plus a reduction request.
 * The wire shape mirrors Rust `goofi_view::ViewSpec`; `ndim` is a conjunction.
 */
import type { ViewerKind } from './kind';
import { VIEWER_KINDS } from '$lib/api/vocab';

export type ReduceMethod = 'envelope' | 'subsample' | 'area';
export type DimCmp = 'lt' | 'le' | 'eq' | 'ge' | 'gt';
export type ViewDtype = 'array' | 'string' | 'table';

export interface AxisReduce {
	dim: number;
	max: number;
	method: ReduceMethod;
}

export interface DimConstraint {
	dim: number;
	cmp: DimCmp;
	n: number;
}

export interface ViewSpec {
	dtype: ViewDtype;
	ndim: [DimCmp, number][];
	dims: DimConstraint[];
	reduce: AxisReduce[];
}

/** Floor so a 0-px / collapsed layout never asks for a degenerate reduction. */
export const CAP_FLOOR = 64;
const MAX_ROWS = 512;
const MAX_POINTS = 4096;

function px(v: number): number {
	return Math.max(CAP_FLOOR, Math.round(v) || CAP_FLOOR);
}

/** The dimension-count constraint a kind declares, read off the manager's `accepts` range —
 * wider than what `line` draws, so a 3-D frame still arrives reduced for HighDimFallback. */
function ndimOf(kind: ViewerKind): [DimCmp, number][] {
	const a = VIEWER_KINDS.find((k) => k.id === kind)?.accepts;
	if (!a) return [];
	if (a[0] === a[1]) return [['eq', a[1]]];
	return a[0] === 0 ? [['le', a[1]]] : [['ge', a[0]], ['le', a[1]]];
}

/** The ViewSpec for a viewer `kind` at `width`×`height` device pixels. */
export function viewSpecForKind(kind: ViewerKind, width: number, height: number): ViewSpec {
	const w = px(width);
	const h = px(height);
	const ndim = ndimOf(kind);
	if (kind === 'line') {
		// For 1-D, dim 0 and -1 collide on the bridge; it resolves by richness (envelope wins).
		return {
			dtype: 'array',
			ndim,
			dims: [],
			reduce: [
				{ dim: 0, max: Math.min(h, MAX_ROWS), method: 'subsample' },
				{ dim: -1, max: w, method: 'envelope' }
			]
		};
	}
	if (kind === 'image') {
		return {
			dtype: 'array',
			ndim,
			dims: [],
			reduce: [
				{ dim: 0, max: h, method: 'area' },
				{ dim: 1, max: w, method: 'area' }
			]
		};
	}
	if (kind === 'trajectory') {
		// (dims × points): the path is the LAST axis, and a phase portrait has no peaks to keep.
		return {
			dtype: 'array',
			ndim,
			dims: [],
			reduce: [{ dim: -1, max: Math.min(w, MAX_POINTS), method: 'subsample' }]
		};
	}
	if (kind === 'topomap') {
		return { dtype: 'array', ndim, dims: [], reduce: [] };
	}
	if (kind === 'string') {
		return { dtype: 'string', ndim, dims: [], reduce: [] };
	}
	return { dtype: 'table', ndim, dims: [], reduce: [] };
}
