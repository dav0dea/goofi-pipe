import { describe, it, expect, beforeEach } from 'vitest';
import { history, type Action } from './history.svelte';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';

const ctx = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };
// A graph history entry marks a step; its undo/redo delegate to the manager (B3).
const mk = (label: string): Action => ({ kind: 'graph_cmd', label, domain: 'graph', context: ctx });

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

	const undoCalls = (fc: FakeControl) => fc.recordedCalls().filter((c) => c.op === 'undo');
	const redoCalls = (fc: FakeControl) => fc.recordedCalls().filter((c) => c.op === 'redo');

	it('undo() fired twice before the first settles delegates to the manager exactly once', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const h = history();
		h.configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		h.record(mk('Set freq'));
		expect(h.canUndo).toBe(true);

		// Two awaits sit between reading the top action and pop(); a held key fires undo() again
		// before the first pops. The guard must drop the second.
		const p1 = h.undo();
		const p2 = h.undo();
		await Promise.all([p1, p2]);

		// The manager undo delegate (control.call('undo')) went out exactly once.
		expect(undoCalls(fc)).toHaveLength(1);
		expect(h.canUndo).toBe(false);
		expect(h.canRedo).toBe(true);

		// And exactly one action round-trips: a single redo empties the redo stack.
		await h.redo();
		expect(redoCalls(fc)).toHaveLength(1);
		expect(h.canRedo).toBe(false);
		expect(h.canUndo).toBe(true);
	});

	it('redo() fired twice before the first settles delegates to the manager exactly once', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const h = history();
		h.configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		h.record(mk('Set freq'));
		await h.undo(); // move the action onto the redo stack

		const p1 = h.redo();
		const p2 = h.redo();
		await Promise.all([p1, p2]);

		// The manager redo delegate went out exactly once — the guard dropped the second redo().
		expect(redoCalls(fc)).toHaveLength(1);
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

describe('HistoryStore — transaction atomicity on throw', () => {
	beforeEach(() => history().reset());

	it('discards the buffered children when fn throws (no orphan undo step)', async () => {
		const h = history();
		await expect(
			h.transaction('Add + boom', async () => {
				h.record(mk('inner add')); // a child gets buffered…
				throw new Error('boom'); // …then the transaction fails partway
			})
		).rejects.toThrow('boom');
		// A failed transaction is not atomic, so it must leave NO undo step behind.
		expect(h.canUndo).toBe(false);
		expect(h.undoLabel).toBe(null);
	});

	it('still commits exactly one undo step when fn succeeds', async () => {
		const h = history();
		await h.transaction('Add', async () => {
			h.record(mk('a'));
			h.record(mk('b'));
		});
		expect(h.canUndo).toBe(true);
		expect(h.undoLabel).toBe('Add'); // two children → one compound under the tx label
	});

	it('folds records that land only AFTER an awaited step (the store records post-RPC)', async () => {
		// A store mutator records its graph_cmd only after its command RPC resolves; the caller
		// (e.g. a multi-node drag transaction) MUST await each mutator so the records land in the
		// buffer before flush. Awaited async records fold into one compound; un-awaited would leak
		// out as separate top-level steps.
		const h = history();
		await h.transaction('Move 2 nodes', async () => {
			for (const label of ['a', 'b']) {
				await Promise.resolve(); // stands in for the awaited command RPC
				h.record(mk(label));
			}
		});
		expect(h.canUndo).toBe(true);
		expect(h.undoLabel).toBe('Move 2 nodes'); // folded, not two separate 'Move' steps
	});
});
