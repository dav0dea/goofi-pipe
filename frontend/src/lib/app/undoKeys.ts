/**
 * Pure decision for the global undo/redo keybindings, factored out of AppShell
 * so it is unit-testable without a DOM. Returns the action a keydown should
 * trigger, or 'none' when it should be ignored (typing in a field, a modal is
 * up, or the combo isn't an undo/redo chord).
 *
 * Bindings: Ctrl/Cmd+Z → undo · Ctrl/Cmd+Shift+Z and Ctrl+Y → redo.
 */
export interface UndoKeyEvent {
	key: string;
	ctrlKey: boolean;
	metaKey: boolean;
	shiftKey: boolean;
	/** Whether the event landed in a text-editing surface — `ui/textEditing.ts`'s answer, which covers
	 *  a contenteditable (X's expression editor) as well as the form elements. */
	editing: boolean;
}

export type UndoKeyResult = 'undo' | 'redo' | 'none';

export function undoKeyAction(e: UndoKeyEvent, modalOpen: boolean): UndoKeyResult {
	if (modalOpen) return 'none';
	if (e.editing) return 'none';
	const meta = e.ctrlKey || e.metaKey;
	if (!meta) return 'none';
	const key = e.key.toLowerCase();
	if (key === 'z') return e.shiftKey ? 'redo' : 'undo';
	if (key === 'y' && !e.shiftKey) return 'redo';
	return 'none';
}
