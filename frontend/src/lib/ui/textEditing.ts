/**
 * "Does this keystroke belong to a text editor, or to the app?" — the one answer both keyboard scopes
 * ask (`app/undoKeys.ts` for undo/redo, `panels/NodeEditorPanel.svelte` for the canvas chords).
 *
 * Both used to spell it inline as `INPUT | TEXTAREA | SELECT`, which was a complete answer for exactly
 * as long as every editable surface in the app was a form element. X's expression editor is a
 * contenteditable `<div>`: under the old spelling its Ctrl+Z would have undone the GRAPH and its
 * Ctrl+A would have selected every node. Unified here rather than patched in both, so the next
 * editable surface cannot be handled in one scope and missed in the other.
 *
 * Lives in `$lib/ui` for the same reason `dragGesture` does — not a primitive, but the shared leaf both
 * layers already import, with no store or app dependency of its own. Duck-typed over the two fields it
 * reads, so it is unit-tested without a DOM.
 */
export function isTextEditingTarget(target: EventTarget | null): boolean {
	const el = target as (Partial<HTMLElement> & EventTarget) | null;
	if (!el) return false;
	if (el.isContentEditable === true) return true;
	const tag = el.tagName ?? '';
	return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}
