/**
 * Per-(node, slot) INLINE viewer view-state: the kind, the cog-menu settings and the collapse of
 * the viewer drawn on the node body.
 *
 * Its ONE owner is the node record's own `viewers` blob, and nothing is stored here — every value
 * below is derived on read from the node it is handed, so a peer's edit, an undo and a re-load all
 * land the moment the record does. For a real node that blob is the DOCUMENT's, projected in by
 * `assembleNode`; for a sub-patch scope it is the instance record's own (see
 * `graph.setSlotView`, which is where a write goes).
 *
 * Docked Viewer PANELS do NOT use this — each panel owns its view state in its own layout state
 * (see viewBinding.panelBinding).
 */
import type { NodeInstanceInfo } from '$lib/api/control';
import type { ViewerKind } from './kind';
import type { SettingsMap } from './settingsSchema';

export interface SlotView {
	collapsed?: boolean;
	kind?: ViewerKind;
	settings?: SettingsMap;
}

/** A slot's stored view, RAW — no dtype resolution, no setting defaults. Empty when unset. */
export function slotView(node: NodeInstanceInfo | null | undefined, slot: string): SlotView {
	return (node?.viewers?.[slot] as SlotView | undefined) ?? {};
}

/** Is a slot's inline viewer open? Absent means the default: visible — except on a node with 3+
 * outputs, which starts collapsed so a fresh spawn cannot bury the canvas under tall viewers. */
export function isSlotExpanded(node: NodeInstanceInfo | null | undefined, slot: string): boolean {
	return !(slotView(node, slot).collapsed ?? Object.keys(node?.output_slots ?? {}).length >= 3);
}
