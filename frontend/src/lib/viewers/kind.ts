/**
 * Viewer-kind vocabulary — the single source of truth for which viewer types
 * exist, which ARRAY kinds the type dropdown offers, and whether a given array
 * shape can be drawn by a given viewer.
 *
 * The dtype→default mapping lives in the `viewerKind` store (STRING/TABLE pin to
 * their dedicated viewers, everything else starts on 'line'); this module just
 * owns the vocabulary the SlotViewer and ViewerPanel share.
 */
import type { ArrayData } from '$lib/codec/decode';

/** The viewer types a slot can be rendered with. */
export type ViewerKind = 'line' | 'image' | 'trajectory' | 'topomap' | 'string' | 'table';

/** ARRAY viewer kinds the viewer-type dropdown offers, in order. */
export const ARRAY_KINDS = ['line', 'image', 'trajectory', 'topomap'] as const;

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
