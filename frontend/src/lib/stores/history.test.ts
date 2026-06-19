import { describe, it, expect, beforeEach } from 'vitest';
import { history, type Action } from './history.svelte';

const ctx = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };
const mk = (label: string): Action => ({
	kind: 'add_node',
	label,
	domain: 'graph',
	context: ctx,
	payload: { type: 'X', category: 'c', pos: [0, 0] }
});

describe('HistoryStore — Phase 1 core', () => {
	beforeEach(() => history().reset());

	it('starts empty: canUndo/canRedo false, labels null', () => {
		const h = history();
		expect([h.canUndo, h.canRedo, h.undoLabel, h.redoLabel]).toEqual([false, false, null, null]);
	});

	it('record() pushes onto the undo stack and exposes the label', () => {
		const h = history();
		h.record(mk('Add Oscillator'));
		expect(h.canUndo).toBe(true);
		expect(h.undoLabel).toBe('Add Oscillator');
		expect(h.canRedo).toBe(false);
	});

	it('record() clears the redo stack and updates the top label', () => {
		const h = history();
		h.record(mk('A'));
		h.record(mk('B'));
		expect(h.canRedo).toBe(false);
		expect(h.undoLabel).toBe('B');
	});

	it('suspend() blocks recording', () => {
		const h = history();
		h.suspend(() => h.record(mk('A')));
		expect(h.canUndo).toBe(false);
	});

	it('suspend() returns the fn value and is reentrant', () => {
		const h = history();
		const r = h.suspend(() => h.suspend(() => 42));
		expect(r).toBe(42);
		expect(h.isSuspended).toBe(false);
	});

	it('reset() clears both stacks', () => {
		const h = history();
		h.record(mk('A'));
		h.reset();
		expect(h.canUndo).toBe(false);
		expect(h.canRedo).toBe(false);
		expect(h.undoLabel).toBe(null);
	});
});
