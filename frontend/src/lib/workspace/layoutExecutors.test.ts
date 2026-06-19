import { describe, it, expect, beforeEach } from 'vitest';
import { workspace } from './workspace.svelte';
import { collectPanels } from './model';
import { layoutExecutors } from './layoutExecutors';
import { history, type Action, type NavContext } from '$lib/stores/history.svelte';

const EMPTY_CTX: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

describe('workspace.restore', () => {
	beforeEach(() => workspace().reset());

	it('assigns a state snapshot exactly, without reseeding panel ids', () => {
		const ws = workspace();
		const before = ws.serialize();
		const beforeIds = collectPanels(ws.active.root).map((p) => p.id);
		ws.split(ws.activePanelId!, 'row');
		expect(collectPanels(ws.active.root).length).toBe(2);
		ws.restore(before);
		const afterIds = collectPanels(ws.active.root).map((p) => p.id);
		expect(afterIds).toEqual(beforeIds); // exact ids, no reseed
		expect(collectPanels(ws.active.root).length).toBe(1);
	});
});

describe('layout executors', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
	});

	it('undo of a split (forward=after / inverse=before) restores the single panel', async () => {
		const ws = workspace();
		const before = ws.serialize();
		ws.split(ws.activePanelId!, 'row');
		const after = ws.serialize();
		const action: Action = {
			kind: 'split_panel',
			label: 'Split panel',
			domain: 'layout',
			context: EMPTY_CTX,
			payload: { before, after }
		};
		const deps = { control: { call: async () => undefined, on: () => () => {}, onConnect: () => () => {} }, graph: {}, workspace: ws } as never;
		await layoutExecutors['split_panel'].inverse(action, deps);
		expect(collectPanels(ws.active.root).length).toBe(1);
		await layoutExecutors['split_panel'].forward(action, deps);
		expect(collectPanels(ws.active.root).length).toBe(2);
	});

	it('undo of close_panel restores the panel with its original id', async () => {
		const ws = workspace();
		ws.split(ws.activePanelId!, 'row');
		const ids = collectPanels(ws.active.root).map((p) => p.id);
		const victim = ids[1];
		const before = ws.serialize();
		ws.close(victim);
		const after = ws.serialize();
		expect(collectPanels(ws.active.root).map((p) => p.id)).not.toContain(victim);
		const action: Action = {
			kind: 'close_panel',
			label: 'Close panel',
			domain: 'layout',
			context: EMPTY_CTX,
			payload: { before, after }
		};
		const deps = { control: { call: async () => undefined, on: () => () => {}, onConnect: () => () => {} }, graph: {}, workspace: ws } as never;
		await layoutExecutors['close_panel'].inverse(action, deps);
		expect(collectPanels(ws.active.root).map((p) => p.id)).toContain(victim);
	});
});

describe('workspace recording wrappers', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
		// Layout undo only needs the workspace store; stub control/graph so the
		// default liveDeps() (which builds a real ControlClient) isn't reached.
		history().configureDeps(() => ({ control: {} as never, graph: {} as never, workspace: workspace() }));
	});

	it('split records a layout action and history.undo reverts it', async () => {
		const ws = workspace();
		ws.split(ws.activePanelId!, 'row');
		expect(history().canUndo).toBe(true);
		expect(collectPanels(ws.active.root).length).toBe(2);
		await history().undo();
		expect(collectPanels(ws.active.root).length).toBe(1);
	});

	it('internal setPanelState does NOT record', () => {
		const ws = workspace();
		ws.setPanelState(ws.activePanelId!, { node: 'osc0' });
		expect(history().canUndo).toBe(false);
	});

	it('linkNodeToPanel DOES record', () => {
		const ws = workspace();
		ws.setType(ws.activePanelId!, 'parameters');
		history().reset();
		ws.linkNodeToPanel(ws.activePanelId!, 'osc0');
		expect(history().canUndo).toBe(true);
		expect(history().undoLabel).toBe('Bind node to panel');
	});

	it('a resize gesture (many resize calls) coalesces into one undo entry', async () => {
		const ws = workspace();
		ws.split(ws.activePanelId!, 'row');
		history().reset();
		// Find the split id + divider.
		const root = ws.active.root;
		const splitId = root.kind === 'split' ? root.id : '';
		for (let i = 0; i < 10; i++) ws.resize(splitId, 0, 0.01);
		// One drag → one entry, not ten.
		await history().undo();
		expect(history().canUndo).toBe(false);
	});

	it('a no-op split (unknown panel) records nothing', () => {
		const ws = workspace();
		ws.split('does-not-exist', 'row');
		expect(history().canUndo).toBe(false);
	});
});
