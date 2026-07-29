/**
 * Imperative handles a node-editor panel exposes to the rest of the app.
 *
 * Several node-editor panels can be open at once, so each registers its handles here keyed by
 * panel id and callers resolve one panel's (falling back to any editor). The readers are the
 * error chip, which focuses an errored node in the editor the user last touched, and the
 * `window.goofi` automation façade, which addresses a panel explicitly.
 */
export interface EditorCommands {
	/** Open the add-node menu centered over this editor. */
	openAddMenu: () => void;
	/** Fit the graph into this editor's viewport. */
	fitView: () => void;
	/** Select/focus a node by name in this editor. */
	focusNode: (name: string) => void;
}

const registry = new Map<string, EditorCommands>();

export function registerEditor(panelId: string, cmds: EditorCommands): void {
	registry.set(panelId, cmds);
}

export function unregisterEditor(panelId: string): void {
	registry.delete(panelId);
}

/** Handles for `panelId` if it's an editor, else the first registered editor,
 * else null when no editor panel is open. */
export function editorFor(panelId: string | null): EditorCommands | null {
	if (panelId && registry.has(panelId)) return registry.get(panelId) ?? null;
	const first = registry.values().next();
	return first.done ? null : first.value;
}
