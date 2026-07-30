import { describe, it, expect } from 'vitest';
import { undoKeyAction, type UndoKeyEvent } from './undoKeys';

const ev = (over: Partial<UndoKeyEvent>): UndoKeyEvent => ({
	key: 'z',
	ctrlKey: false,
	metaKey: false,
	shiftKey: false,
	editing: false,
	...over
});

describe('undoKeyAction', () => {
	it('Ctrl+Z → undo', () => {
		expect(undoKeyAction(ev({ ctrlKey: true, key: 'z' }), false)).toBe('undo');
	});
	it('Cmd+Z → undo', () => {
		expect(undoKeyAction(ev({ metaKey: true, key: 'z' }), false)).toBe('undo');
	});
	it('Ctrl+Shift+Z → redo', () => {
		expect(undoKeyAction(ev({ ctrlKey: true, shiftKey: true, key: 'z' }), false)).toBe('redo');
	});
	it('Ctrl+Y → redo', () => {
		expect(undoKeyAction(ev({ ctrlKey: true, key: 'y' }), false)).toBe('redo');
	});
	it('plain z → none', () => {
		expect(undoKeyAction(ev({ key: 'z' }), false)).toBe('none');
	});
	// `editing` is answered by `isTextEditingTarget`, which since X covers a contenteditable too — the
	// expression editor's Ctrl+Z has to reach CodeMirror's history, not the graph's (D-X7).
	it('suppressed while typing in any text-editing surface', () => {
		expect(undoKeyAction(ev({ ctrlKey: true, key: 'z', editing: true }), false)).toBe('none');
	});
	it('suppressed while a modal is open', () => {
		expect(undoKeyAction(ev({ ctrlKey: true, key: 'z' }), true)).toBe('none');
	});
});
