/**
 * Per-(node, slot) viewer kind for the in-canvas SlotViewers.
 *
 * The chosen viewer type used to live as component-local `$state` inside
 * SlotViewer, reachable only by clicking the cycle button. Hoisting it into a
 * keyed store makes it addressable: the cycle button, a restored patch, and the
 * agent command surface all read/write the same place, and the canvas + any
 * other view of the same slot stay in sync.
 *
 * STRING / TABLE slots always resolve to their dedicated viewer regardless of
 * any stored value, so a stale array-kind never leaks onto a non-array slot
 * (this also subsumes the old "reset kind when dtype changes" effect).
 */
import { cycleKind, type ViewerKind } from './kind';

function key(node: string, slot: string): string {
	return `${node}|${slot}`;
}

const kinds = $state<Record<string, ViewerKind>>({});

/** The viewer kind currently chosen for a slot, given its dtype. */
export function viewerKind(node: string, slot: string, dtype: string): ViewerKind {
	if (dtype === 'STRING') return 'string';
	if (dtype === 'TABLE') return 'table';
	return kinds[key(node, slot)] ?? 'line';
}

/** Set the viewer kind for a slot. */
export function setViewerKind(node: string, slot: string, kind: ViewerKind): void {
	kinds[key(node, slot)] = kind;
}

/** Advance a slot's ARRAY viewer to the next kind in the cycle. */
export function cycleViewerKind(node: string, slot: string, dtype: string): void {
	setViewerKind(node, slot, cycleKind(viewerKind(node, slot, dtype)));
}
