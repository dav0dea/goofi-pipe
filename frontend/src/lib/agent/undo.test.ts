import { describe, it, expect, beforeEach } from 'vitest';
import { commands } from './commands';
import { query } from './query';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from '$lib/stores/graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { history, type Action, type NavContext } from '$lib/stores/history.svelte';

const ctx: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };
const mk = (label: string): Action => ({
	kind: 'add_node',
	label,
	domain: 'graph',
	context: ctx,
	payload: { type: 'X', category: 'c', pos: [0, 0], assignedName: 'x0' }
});

describe('agent surface — undo/redo', () => {
	beforeEach(() => {
		history().reset();
		const fc = new FakeControl();
		history().configureDeps(() => ({ control: fc, graph: new GraphStore(fc), workspace: workspace() }));
	});

	it('query.canUndo / canRedo / historyLength reflect the history store', () => {
		expect(query.canUndo()).toBe(false);
		history().record(mk('A'));
		expect(query.canUndo()).toBe(true);
		expect(query.historyLength()).toBe(1);
		expect(query.undoLabel()).toBe('A');
	});

	it('commands.undo / redo drive the history store', async () => {
		history().record(mk('A'));
		await commands.undo();
		expect(query.canUndo()).toBe(false);
		expect(query.canRedo()).toBe(true);
		await commands.redo();
		expect(query.canUndo()).toBe(true);
	});
});
