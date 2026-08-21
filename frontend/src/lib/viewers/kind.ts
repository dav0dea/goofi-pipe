/** Viewer-kind BEHAVIOUR; the vocabulary itself is the manager's, in `$lib/api/vocab`. */
import type { ArrayData } from '$lib/codec/decode';
import { VIEWER_KINDS, type ViewerKind } from '$lib/api/vocab';

export type { ViewerKind };

/** ARRAY viewer kinds the viewer-type dropdown offers, in order. */
export const ARRAY_KINDS: readonly ViewerKind[] = VIEWER_KINDS.filter(
	(k) => k.dtype === 'ARRAY'
).map((k) => k.id);

/** The viewer kind to actually use: a dtype-pinned kind wins over the stored one. */
export function resolveKind(dtype: string | null, stored: ViewerKind | undefined): ViewerKind {
	const pinned = VIEWER_KINDS.find((k) => k.dtype !== 'ARRAY' && k.dtype === dtype);
	return pinned ? pinned.id : (stored ?? 'line');
}

/** Whether an array of the given shape can be drawn by `kind`; a non-array frame always can. */
export function isRenderable(kind: ViewerKind, spec: ArrayData | null): boolean {
	if (!spec) return true;
	const s = spec.shape;
	const draws = VIEWER_KINDS.find((k) => k.id === kind)?.draws;
	if (!draws) return true;
	if (s.length < draws[0] || s.length > draws[1]) return false;
	if (kind === 'image') return s.length === 2 || [1, 2, 3, 4].includes(s[2]);
	if (kind === 'trajectory') return s[0] >= 2;
	return true;
}
