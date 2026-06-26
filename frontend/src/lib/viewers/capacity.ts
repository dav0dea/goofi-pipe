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
		// 1-D data: axis 0 and -1 both canonicalize to axis 0; the node resolves the
		// collision by richness (envelope > subsample), so the waveform keeps its peaks.
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

/** Richer method wins when two viewers fold onto the same axis. Mirrors
 * `node_reduce._RICHNESS` so the in-tab fold agrees with the node/bridge fold. */
const RICHNESS: Record<ReduceMethod, number> = { envelope: 3, area: 2, subsample: 1 };

/**
 * Fold several viewers' ViewSpecs into one, richest-wins per axis (max of the
 * per-axis `max`, richest method). Used to combine multiple viewers of the SAME
 * (node, slot, kind) within one tab — they share a single WS, so the node must
 * reduce to the union of what they can show (the bridge folds across tabs too).
 *
 * Mirrors the manager-side Python `node_reduce.fold_viewspecs` (which routes the
 * per-axis collision through `_fold_axis`). The shared rule is pinned by the
 * cross-language golden in tests/viewspec_golden.json — see capacity.test.ts.
 */
export function foldViewSpecs(specs: ViewSpec[]): ViewSpec {
	const byAxis = new Map<number, AxisSpec>();
	let version = 0;
	for (const s of specs) {
		if (!s) continue;
		version = Math.max(version, s.version || 0);
		for (const a of s.axes ?? []) {
			const cur = byAxis.get(a.axis);
			if (!cur) {
				byAxis.set(a.axis, { axis: a.axis, max: a.max, method: a.method });
			} else {
				cur.max = Math.max(cur.max, a.max);
				if ((RICHNESS[a.method] ?? 0) > (RICHNESS[cur.method] ?? 0)) cur.method = a.method;
			}
		}
	}
	const axes = [...byAxis.keys()].sort((x, y) => x - y).map((k) => byAxis.get(k) as AxisSpec);
	return { axes, version };
}
