/**
 * Viewer-kind vocabulary — the single source of truth for which viewer types
 * exist, what a slot defaults to, how the cycle button rotates, and whether a
 * given array shape can be drawn by a given viewer.
 *
 * Previously this logic was re-implemented (and had drifted) across SlotViewer,
 * ViewerBody, and ViewerPanel; centralizing it also gives the agent surface one
 * place to answer "what can this slot be shown as".
 */
import type { ArrayData } from '$lib/codec/decode';

/** The viewer types a slot can be rendered with. */
export type ViewerKind = 'line' | 'image' | 'trajectory' | 'topomap' | 'string' | 'table';

/** ARRAY viewer kinds the cycle button rotates through, in order. */
export const ARRAY_KINDS = ['line', 'image', 'trajectory', 'topomap'] as const;

/** The natural default viewer for a slot of the given dtype. */
export function defaultKind(dtype: string | null): ViewerKind {
	if (dtype === 'STRING') return 'string';
	if (dtype === 'TABLE') return 'table';
	return 'line';
}

/** Next ARRAY viewer kind after `kind` (wraps). Non-array kinds are returned
 * unchanged — only ARRAY slots cycle. */
export function cycleKind(kind: ViewerKind): ViewerKind {
	const i = ARRAY_KINDS.indexOf(kind as (typeof ARRAY_KINDS)[number]);
	if (i < 0) return kind;
	return ARRAY_KINDS[(i + 1) % ARRAY_KINDS.length];
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
