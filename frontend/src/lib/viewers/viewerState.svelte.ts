/**
 * Per-(node, slot) viewer kind, shared by the in-canvas SlotViewer and the
 * docked ViewerPanel.
 *
 * The chosen viewer type used to live as component-local `$state` inside
 * SlotViewer. Hoisting it into a keyed store makes it addressable: the type
 * dropdown, a restored patch, the agent command surface, and the panel all
 * read/write the same place, so every view of a slot stays in sync.
 *
 * STRING / TABLE slots always resolve to their dedicated viewer regardless of
 * any stored value, so a stale array-kind never leaks onto a non-array slot
 * (this also subsumes the old "reset kind when dtype changes" effect).
 */
import type { ViewerKind } from './kind';

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

/** Seed a slot's kind from a restored patch (no-op when absent). */
export function seedViewerKind(node: string, slot: string, kind: ViewerKind | undefined): void {
	if (kind) kinds[key(node, slot)] = kind;
}

/** Drop the stored kinds for a node that no longer exists. */
export function forgetViewerKinds(node: string): void {
	const prefix = `${node}|`;
	for (const k of Object.keys(kinds)) if (k.startsWith(prefix)) delete kinds[k];
}
