/**
 * Per-(node, slot) INLINE viewer view-state, derived on read from the node's `viewers` blob.
 * Docked viewer PANELS do not use this — see `viewBinding.panelBinding`.
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

/** Is a slot's inline viewer open? Default visible, but collapsed on a node with 3+ outputs. */
export function isSlotExpanded(node: NodeInstanceInfo | null | undefined, slot: string): boolean {
	return !(slotView(node, slot).collapsed ?? Object.keys(node?.output_slots ?? {}).length >= 3);
}
