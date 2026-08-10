/**
 * Viewer-kind BEHAVIOUR. The vocabulary itself — which kinds exist, which dtype
 * each serves and how many dimensions each takes — is the manager's, declared in
 * `crates/goofi-bridge/src/vocab.rs` and generated into `$lib/api/vocab`; this
 * module is what the app DOES with it, which is the half that depends on a live
 * frame and therefore stays here.
 *
 * The dtype→kind resolution lives in `resolveKind` (a STRING/TABLE slot pins its
 * own viewer, everything else uses the stored kind, default 'line'); every viewer
 * instance resolves through it.
 */
import type { ArrayData } from '$lib/codec/decode';
import { VIEWER_KINDS, type ViewerKind } from '$lib/api/vocab';

export type { ViewerKind };

/** ARRAY viewer kinds the viewer-type dropdown offers, in order. Derived, not
 * listed a second time: a kind pinned to STRING/TABLE is chosen by the slot's
 * dtype, so it is never something to offer. */
export const ARRAY_KINDS: readonly ViewerKind[] = VIEWER_KINDS.filter(
	(k) => k.dtype === 'ARRAY'
).map((k) => k.id);

/** The viewer kind to actually use: a slot whose dtype PINS a kind gets it;
 * ARRAY (and anything else) uses the stored kind, defaulting to line. */
export function resolveKind(dtype: string | null, stored: ViewerKind | undefined): ViewerKind {
	const pinned = VIEWER_KINDS.find((k) => k.dtype !== 'ARRAY' && k.dtype === dtype);
	return pinned ? pinned.id : (stored ?? 'line');
}

/** Whether an array of the given shape can be drawn by `kind`. A non-array
 * frame (no spec) is always renderable by its own dedicated viewer. */
export function isRenderable(kind: ViewerKind, spec: ArrayData | null): boolean {
	if (!spec) return true;
	const s = spec.shape;
	const draws = VIEWER_KINDS.find((k) => k.id === kind)?.draws;
	// A pinned kind draws its own dtype whatever the shape; an unknown one is not this
	// module's to refuse (the manager already did).
	if (!draws) return true;
	// The dimension COUNT is the vocabulary's — `capacity.ts` reads the same numbers, so
	// "compatible for the merge" and "the component will actually draw it" can no longer
	// drift apart. What each viewer does with the axes stays here.
	if (s.length < draws[0] || s.length > draws[1]) return false;
	if (kind === 'image') return s.length === 2 || [1, 2, 3, 4].includes(s[2]);
	if (kind === 'trajectory') return s[0] >= 2;
	return true;
}
