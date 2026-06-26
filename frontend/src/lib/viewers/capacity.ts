/**
 * Capacity → ViewSpec: derive the per-axis reduction a viewer can actually
 * display from its pixel size + kind (Option C node-side reduction). The browser
 * sends this spec to the node (via the /data WS side-channel); the node reduces
 * each frame to it before sending, so a 44.1 kHz buffer ships ~2·width points
 * instead of millions. Pure + unit-tested; the Svelte glue (ViewerFeed) measures
 * the element and calls this.
 *
 * Mirrors the per-kind axis table in node_reduce.py (§6.3): line → envelope on
 * the sample axis (+ subsample the channel axis), image → area on both pixel
 * axes, trajectory → subsample the point axis, everything else → no reduction.
 */
import type { ViewerKind } from './kind';

export type ReduceMethod = 'envelope' | 'subsample' | 'area';

export interface AxisSpec {
	axis: number;
	max: number;
	method: ReduceMethod;
}

export interface ViewSpec {
	axes: AxisSpec[];
	version: number;
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

/**
 * The ViewSpec for a viewer `kind` at `width`×`height` device pixels.
 * `version` is a client-side ordering counter (the node ignores it).
 */
export function viewSpecForKind(
	kind: ViewerKind,
	width: number,
	height: number,
	version = 0
): ViewSpec {
	const w = px(width);
	const h = px(height);
	let axes: AxisSpec[];
	if (kind === 'line') {
		// 1-D data: axis 0 and -1 both canonicalize to axis 0, last-wins → envelope.
		// 2-D (C,N): axis 0 caps channels (subsample), axis -1 envelopes the samples.
		axes = [
			{ axis: 0, max: Math.min(h, MAX_ROWS), method: 'subsample' },
			{ axis: -1, max: w, method: 'envelope' }
		];
	} else if (kind === 'image') {
		axes = [
			{ axis: 0, max: h, method: 'area' },
			{ axis: 1, max: w, method: 'area' }
		];
	} else if (kind === 'trajectory') {
		axes = [{ axis: 0, max: Math.min(w, MAX_POINTS), method: 'subsample' }];
	} else {
		// topomap / string / table → already tiny or non-array; no reduction.
		axes = [];
	}
	return { axes, version };
}
