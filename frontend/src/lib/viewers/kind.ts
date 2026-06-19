/**
 * Viewer-kind vocabulary — the single source of truth for which viewer types
 * exist, which ARRAY kinds the type dropdown offers, and whether a given array
 * shape can be drawn by a given viewer.
 *
 * The dtype→kind resolution lives in `resolveKind` here (STRING/TABLE pin to
 * their dedicated viewers, everything else uses the stored kind, default 'line');
 * every viewer instance resolves through it.
 */
import type { ArrayData } from '$lib/codec/decode';

/** The viewer types a slot can be rendered with. */
export type ViewerKind = 'line' | 'image' | 'trajectory' | 'topomap' | 'string' | 'table';

/** ARRAY viewer kinds the viewer-type dropdown offers, in order. */
export const ARRAY_KINDS = ['line', 'image', 'trajectory', 'topomap'] as const;

/** The viewer kind to actually use: STRING/TABLE slots force their dedicated
 * viewer; ARRAY (and anything else) uses the stored kind, defaulting to line. */
export function resolveKind(dtype: string | null, stored: ViewerKind | undefined): ViewerKind {
	if (dtype === 'STRING') return 'string';
	if (dtype === 'TABLE') return 'table';
	return stored ?? 'line';
}

/** Whether an array of the given shape can be drawn by `kind`. A non-array
 * frame (no spec) is always renderable by its own dedicated viewer. */
export function isRenderable(kind: ViewerKind, spec: ArrayData | null): boolean {
	if (!spec) return true;
	const s = spec.shape;
	if (kind === 'line') return s.length <= 3;
	if (kind === 'image') return s.length === 2 || (s.length === 3 && [1, 2, 3, 4].includes(s[2]));
	if (kind === 'trajectory') return s.length === 2 && s[0] >= 2;
	if (kind === 'topomap') return s.length === 1;
	return true;
}
