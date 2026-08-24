/** Imperative handles a node-editor panel exposes to the rest of the app. */
export interface EditorCommands {
	openAddMenu: () => void;
	fitView: () => void;
	focusNode: (name: string) => void;
	selectAll: () => void;
	clearSelection: () => void;
	deleteSelection: () => void;
	groupSelection: () => void;
	copySelection: () => void;
	cutSelection: () => void;
	pasteClipboard: () => void;
	duplicateSelection: () => void;
	/** Store selection AND a live marquee. */
	hasSelection: () => boolean;
}

const registry = new Map<string, EditorCommands>();

export function registerEditor(panelId: string, cmds: EditorCommands): void {
	registry.set(panelId, cmds);
}

export function unregisterEditor(panelId: string): void {
	registry.delete(panelId);
}

/** Handles for `panelId` if it's an editor, else the first registered editor. */
export function editorFor(panelId: string | null): EditorCommands | null {
	if (panelId && registry.has(panelId)) return registry.get(panelId) ?? null;
	const first = registry.values().next();
	return first.done ? null : first.value;
}

/** Handles for `panelId` if and only if it is a live editor panel. No fallback. */
export function editorAt(panelId: string | null): EditorCommands | null {
	return panelId ? (registry.get(panelId) ?? null) : null;
}

/** The editor an app-global command means: the active one, or the only one; never a guess. */
export function activeOrOnlyEditor(panelId: string | null): EditorCommands | null {
	return editorAt(panelId) ?? (registry.size === 1 ? (registry.values().next().value ?? null) : null);
}
