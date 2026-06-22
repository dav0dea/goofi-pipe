/**
 * Reading the bridge viewer-adapter's `meta["__view__"]` block (viewer-adapters
 * design, backlog #3). Adapted frames carry float range/stats here so the viewer's
 * value range and the metadata inspector stay float-accurate even though the wire
 * body is uint8/float16. All readers are defensive: an absent `__view__` (string/
 * table/raw frames, or a pre-adapter producer) falls back to the received array.
 */
import type { ArrayData } from '$lib/codec/decode';

export interface ViewSummary {
	shape: number[];
	dtype: string;
	min: number | null;
	mean: number | null;
	max: number | null;
}

export interface ViewInfo {
	/** Image only: original float [fmin, fmax] of the source array. */
	range?: [number, number];
	/** Array kinds: exact float stats (pre-downcast). */
	stats?: { min: number; mean: number; max: number };
	/** Non-renderable frames: a bodyless float summary. */
	summary?: ViewSummary;
}

/** Read `meta["__view__"]` defensively. */
export function viewInfo(meta: Record<string, unknown> | undefined): ViewInfo {
	const v = meta?.['__view__'];
	if (!v || typeof v !== 'object') return {};
	return v as ViewInfo;
}

/** The shape/dtype/stats to show for a non-renderable frame: the backend float
 * summary if present, else computed from the received array values. */
export function summaryOf(arraySpec: ArrayData, view: ViewInfo): ViewSummary {
	if (view.summary) return view.summary;
	const v = arraySpec.values;
	let mn = Infinity;
	let mx = -Infinity;
	let sum = 0;
	let n = 0;
	for (let i = 0; i < v.length; i++) {
		const x = Number(v[i]);
		if (!Number.isFinite(x)) continue;
		if (x < mn) mn = x;
		if (x > mx) mx = x;
		sum += x;
		n++;
	}
	return {
		shape: arraySpec.shape,
		dtype: arraySpec.dtype,
		min: n ? mn : null,
		mean: n ? sum / n : null,
		max: n ? mx : null
	};
}

function isUint8(dtype: string): boolean {
	return dtype.endsWith('u1');
}

/** Map a manual gray window from float units into the array's data domain. When
 * the wire body is uint8 (image adapter), the source float range lives in
 * `view.range`; a float window [vmin,vmax] maps to uint8 via that range. Float
 * bodies (no adapter, or pre-adapter) pass through unchanged. */
export function manualGrayDomain(
	view: ViewInfo,
	vmin: number,
	vmax: number,
	dtype: string
): [number, number] {
	if (isUint8(dtype) && view.range) {
		const [fmin, fmax] = view.range;
		const span = fmax - fmin || 1e-9;
		return [((vmin - fmin) / span) * 255, ((vmax - fmin) / span) * 255];
	}
	return [vmin, vmax];
}
