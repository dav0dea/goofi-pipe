/**
 * Per-(node, slot) INLINE viewer view-state: the kind + settings of the viewer
 * shown on the node body. One entry per slot; seeded from node.viewers on load,
 * pushed back (debounced) via graph.pushNodeViewers. Kept separate from the node
 * DATA record so a node state-update / snapshot replacement never clobbers live
 * view edits. Docked Viewer PANELS do NOT use this — each panel owns its view
 * state in its own layout state (see viewBinding.panelBinding).
 */
import type { ViewerKind } from './kind';
import type { SettingValue, SettingsMap } from './settingsSchema';

interface InlineView {
	kind?: ViewerKind;
	settings: SettingsMap;
}

function key(node: string, slot: string): string {
	return `${node}|${slot}`;
}

function emptyView(): InlineView {
	return { settings: {} };
}

const store = $state<Record<string, InlineView>>({});

/** Raw stored inline view for a slot (no defaults applied). */
export function rawInlineView(node: string, slot: string): InlineView {
	return store[key(node, slot)] ?? emptyView();
}

export function setInlineKind(node: string, slot: string, kind: ViewerKind): void {
	const id = key(node, slot);
	store[id] = { ...(store[id] ?? emptyView()), kind };
}

export function setInlineSetting(node: string, slot: string, k: string, value: SettingValue): void {
	const id = key(node, slot);
	const cur = store[id] ?? emptyView();
	store[id] = { ...cur, settings: { ...cur.settings, [k]: value } };
}

/** Seed a slot's inline view from a restored patch (no-op when empty). */
/** Replace a slot's whole inline view (kind + settings) — used by undo/redo to
 * restore a captured snapshot. */
export function setInlineFullView(
	node: string,
	slot: string,
	view: { kind?: ViewerKind; settings: SettingsMap }
): void {
	store[key(node, slot)] = { kind: view.kind, settings: { ...view.settings } };
}

export function seedInlineView(
	node: string,
	slot: string,
	view: { kind?: ViewerKind; settings?: SettingsMap } | undefined
): void {
	if (!view) return;
	const hasKind = view.kind != null;
	const hasSettings = view.settings && Object.keys(view.settings).length > 0;
	if (!hasKind && !hasSettings) return;
	store[key(node, slot)] = { kind: view.kind, settings: { ...(view.settings ?? {}) } };
}

/** Drop every slot's inline view for a node that no longer exists. */
export function forgetInlineView(node: string): void {
	const prefix = `${node}|`;
	for (const k of Object.keys(store)) if (k.startsWith(prefix)) delete store[k];
}
