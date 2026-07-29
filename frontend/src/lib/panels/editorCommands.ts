/**
 * Imperative handles a node-editor panel exposes to the rest of the app.
 *
 * Several node-editor panels can be open at once, so each registers its handles here keyed by
 * panel id, and there are two ways to resolve one — deliberately, because the callers want
 * different things when the id names no editor:
 *
 * - `editorFor` is LENIENT (falls back to any editor). Its readers are the error chip, which is
 *   showing you an errored node and just needs somewhere to show it, and the `window.goofi`
 *   automation façade, which names a panel explicitly.
 * - `editorAt` is STRICT. The app header addresses through it: a Delete or Group launched from
 *   app-global chrome must land in the editor the user is actually working in, never in whichever
 *   one happens to be first in the map. With no unambiguous target the header disables the row
 *   rather than guessing (R-Task 6).
 */
export interface EditorCommands {
	/** Open the add-node menu centered over this editor. */
	openAddMenu: () => void;
	/** Fit the graph into this editor's viewport. */
	fitView: () => void;
	/** Select/focus a node by name in this editor. */
	focusNode: (name: string) => void;
	/** Select everything in the scope this editor is showing (⌘A's action). */
	selectAll: () => void;
	/** Delete this editor's selected nodes and edges (the Delete key's action). */
	deleteSelection: () => void;
	/** Collapse this editor's selected nodes into a sub-patch (⌘G's action). */
	groupSelection: () => void;
	/** Copy / paste / duplicate the selection (⌘C / ⌘V / ⌘D). */
	copySelection: () => void;
	pasteClipboard: () => void;
	duplicateSelection: () => void;
	/** Whether anything is selected right now — store selection AND a live marquee. */
	hasSelection: () => boolean;
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

/** Handles for `panelId` if and only if it is a live editor panel. No fallback. */
export function editorAt(panelId: string | null): EditorCommands | null {
	return panelId ? (registry.get(panelId) ?? null) : null;
}
