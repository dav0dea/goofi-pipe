/**
 * Capacity → ViewSpec: the standardized per-viewer declaration the browser sends
 * to the bridge on the /data socket — a *compatibility predicate* (what the viewer
 * can draw) plus a *reduction request* (what it wants the drawable axes shrunk to).
 *
 * Every viewer of a (node, slot) contributes ONE ViewSpec; the bridge merges them
 * (`goofi_view::plan`) against the actual frame — incompatible viewers drop out,
 * the rest fold to the largest need per axis — and reduces the slot ONCE
 * (`reduce_for_view`), so a 44.1 kHz buffer ships ~2·width points instead of
 * millions. The fold lives on the bridge now (it needs the real frame to filter
 * compatibility), so this module only *produces* per-kind specs. Pure + unit-tested.
 *
 * Wire shape mirrors Rust `goofi_view::ViewSpec` exactly (snake_case enums):
 *   { dtype, ndim: [[cmp, n], …], dims: [{dim, cmp, n}], reduce: [{dim, max, method}] }
 * The `ndim` list is a conjunction (ALL must hold) so a viewer can bound a range —
 * image is 2-D or 3-D → [['ge',2],['le',3]].
 */
import type { ViewerKind } from './kind';

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
/** Cap on channels (rows) a line plot subsamples to — more than this is unreadable. */
const MAX_ROWS = 512;
/** Cap on trajectory points. */
const MAX_POINTS = 4096;

function px(v: number): number {
	return Math.max(CAP_FLOOR, Math.round(v) || CAP_FLOOR);
}

/** The ViewSpec for a viewer `kind` at `width`×`height` device pixels. The `ndim`
 * range mirrors `isRenderable` in kind.ts so "compatible for the merge" agrees with
 * "the component will actually draw it" — with one deliberate exception: line accepts
 * `le 3` here but only draws `le 2`, so a 3-D frame is still reduced for the viewer
 * that shows its HighDimFallback summary rather than shipped at full size. */
export function viewSpecForKind(kind: ViewerKind, width: number, height: number): ViewSpec {
	const w = px(width);
	const h = px(height);
	if (kind === 'line') {
		// 1-D data: dim 0 and -1 both canonicalize to dim 0 on the bridge; it resolves
		// the collision by richness (envelope > subsample), so the waveform keeps its
		// peaks. 2-D (C,N): dim 0 caps channels (subsample), dim -1 envelopes the samples.
		return {
			dtype: 'array',
			ndim: [['le', 3]],
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
			ndim: [
				['ge', 2],
				['le', 3]
			],
			dims: [],
			reduce: [
				{ dim: 0, max: h, method: 'area' },
				{ dim: 1, max: w, method: 'area' }
			]
		};
	}
	if (kind === 'trajectory') {
		// A trajectory frame is (dims × points): TrajectoryViewer pairs ROWS i<j and walks
		// shape[1] as the path, so the long axis — the one worth reducing — is the LAST one.
		// Subsample, not envelope: a phase portrait has no min/max pairing to preserve, and
		// the viewer ignores `meta.reduced`. The rows are left alone; capping them would drop
		// whole signals, and a row count is small by construction.
		return {
			dtype: 'array',
			ndim: [['eq', 2]],
			dims: [],
			reduce: [{ dim: -1, max: Math.min(w, MAX_POINTS), method: 'subsample' }]
		};
	}
	if (kind === 'topomap') {
		// Per-channel scalars — already tiny; declare the shape, request no reduction.
		return { dtype: 'array', ndim: [['eq', 1]], dims: [], reduce: [] };
	}
	if (kind === 'string') {
		return { dtype: 'string', ndim: [], dims: [], reduce: [] };
	}
	// table
	return { dtype: 'table', ndim: [], dims: [], reduce: [] };
}
