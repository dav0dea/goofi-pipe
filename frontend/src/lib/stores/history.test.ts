import { describe, it, expect, beforeEach } from 'vitest';
import { history, type Action } from './history.svelte';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';

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

describe('HistoryStore — re-entrancy (report B13: held Ctrl+Z)', () => {
	beforeEach(() => history().reset());

	const paramAction = (): Action => ({
		kind: 'update_param',
		label: 'Set freq',
		domain: 'graph',
		context: ctx,
		payload: { node: 'osc0', group: 'common', name: 'frequency', oldValue: 1, newValue: 5 }
	});

	it('undo() fired twice before the first settles replays the inverse exactly once', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const h = history();
		h.configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		h.record(paramAction());
		expect(h.canUndo).toBe(true);

		// Two awaits sit between reading the top action and pop(); a held key
		// fires undo() again before the first pops. The guard must drop it.
		const p1 = h.undo();
		const p2 = h.undo();
		await Promise.all([p1, p2]);

		const inverseCalls = fc.recordedCalls().filter((c) => c.op === 'update_param');
		expect(inverseCalls).toHaveLength(1); // inverse ran once, not twice
		expect(h.canUndo).toBe(false);
		expect(h.canRedo).toBe(true);

		// And exactly one action round-trips: a single redo empties the redo stack.
		await h.redo();
		expect(h.canRedo).toBe(false);
		expect(h.canUndo).toBe(true);
	});

	it('redo() fired twice before the first settles replays the forward exactly once', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const h = history();
		h.configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		h.record(paramAction());
		await h.undo(); // move the action onto the redo stack
		fc.recordedCalls().length = 0;

		const p1 = h.redo();
		const p2 = h.redo();
		await Promise.all([p1, p2]);

		const forwardCalls = fc.recordedCalls().filter((c) => c.op === 'update_param');
		expect(forwardCalls).toHaveLength(1);
		expect(h.canRedo).toBe(false);
		expect(h.canUndo).toBe(true);
	});
});

describe('HistoryStore — lastError surface (#9)', () => {
	beforeEach(() => history().reset());

	it('clearError() resets lastError so a dismissed toast stays dismissed', () => {
		const h = history();
		h.lastError = 'Undo failed: name taken';
		h.clearError();
		expect(h.lastError).toBe(null);
	});
});
