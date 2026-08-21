/** The global undo/redo chords: Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z and Ctrl+Y redo. */
export interface UndoKeyEvent {
	key: string;
	ctrlKey: boolean;
	metaKey: boolean;
	shiftKey: boolean;
	/** Whether the event landed in a text-editing surface, a contenteditable included. */
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
