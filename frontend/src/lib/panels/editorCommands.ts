/**
 * Imperative handles a node-editor panel exposes to the rest of the app.
 *
 * The global TopBar has "Add node" and "Fit" buttons, but there can be several
 * node-editor panels open at once. Each registers its handles here keyed by
 * panel id; the TopBar resolves the active panel's handles (falling back to any
 * editor) so its buttons drive whichever editor the user last touched.
 */
export interface EditorCommands {
	/** Open the add-node menu centered over this editor. */
	openAddMenu: () => void;
	/** Fit the graph into this editor's viewport. */
	fitView: () => void;
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
