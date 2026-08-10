import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { nodesMap, instancesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import * as Y from 'yjs';

/** Minimal catalog — its presence flips the store to doc-authoritative identity. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Buffer',
			category: 'signal',
			doc: '',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			params: {}
		}
	];
}

/** A hello/graph_replaced snapshot — session frame + runtime overlay only (structure is the doc's). */
function snapshot(): GraphSnapshot {
	return {
		runtime: {},
		save_path: null,
		unsaved_changes: false,
		instance_id: 'sess1',
		layout: null
	} as never;
}

/** A store holding a collapsed sub-patch instance `sub` (one member `m1`). */
function withInstance(): { fc: FakeControl; g: GraphStore } {
	const fc = new FakeControl();
	const g = new GraphStore(fc);
	g.nodeTypes = catalog();
	// Seed the doc exactly as the manager's mirror writes it — the scope forest's single source.
	Y.transact(g.doc, () => {
		const n = new Y.Map<unknown>();
		n.set('type', 'Buffer');
		n.set('name', 'buffer0');
		nodesMap(g.doc).set('m1', n);

		const inst = new Y.Map<unknown>();
		inst.set('name', 'subpatch0');
		inst.set('parent', ROOT_ID);
		const members = new Y.Map<Y.Map<unknown>>();
		const entry = new Y.Map<unknown>();
		entry.set('is_instance', false);
		members.set('m1', entry);
		inst.set('members', members);
		instancesMap(g.doc).set('sub', inst);
	});
	return { fc, g };
}

describe('a wholesale load resets the client history (lockstep with the manager)', () => {
	beforeEach(() => history().reset());
	const ctx = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

	it('graph_replaced (in-session load) drops pre-load undo steps', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		// Establish the session, then record a pre-load graph step.
		fc.emit({ event: 'hello', payload: snapshot() });
		history().record({ kind: 'graph_cmd', domain: 'graph', label: 'Add X', context: ctx });
		expect(history().canUndo).toBe(true);

		// A load replaces the graph in the SAME backend session — the manager cleared its command
		// history (`load`/`new` → CommandHistory::clear), so the client's stale entries would pop
		// mismatched. graph_replaced must reset the client history too.
		fc.emit({ event: 'graph_replaced', payload: snapshot() });
		expect(history().canUndo).toBe(false);
		void g;
	});
});

describe('undo of a delete re-binds panels the delete emptied', () => {
	beforeEach(() => {
		history().reset();
		workspace().reset();
	});

	it('captures bound panels on removeNode and re-binds them on undo', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const ws = workspace();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: ws }));

		// Bind a Parameters panel to the node, then isolate the delete (drop the layout entry).
		const panelId = ws.activePanelId!;
		ws.setType(panelId, 'parameters');
		ws.linkNodeToPanel(panelId, 'osc0');
		expect(ws.panelsBoundTo('osc0').map((p) => p.panelId)).toContain(panelId);
		history().reset();

		// Delete: removeNode captures the bound panels BEFORE the RPC; the doc-reconcile then empties
		// them when the node vanishes (simulated here — FakeControl doesn't mutate the doc).
		await g.removeNode('osc0');
		ws.clearNodeRefs('osc0');
		expect(ws.panelsBoundTo('osc0')).toHaveLength(0);

		// Undo delegates to the manager AND re-binds the emptied panel (the graph executor's only
		// client-local side-effect).
		await history().undo();
		expect(ws.panelsBoundTo('osc0').map((p) => p.panelId)).toContain(panelId);
	});

	// The capture (`panelsBoundTo`) and the clearing (`clearNodeRefs`) both walk EVERY tab, so the
	// restore has to as well — otherwise a panel bound in a background tab is emptied by the delete
	// and never re-bound, and the undo silently loses it.
	it('re-binds a panel that lives in a BACKGROUND layout tab', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const ws = workspace();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: ws }));

		// Bind a Parameters panel in a second tab, then leave that tab.
		const firstTab = ws.state.activeWorkspaceId;
		ws.addTab('parameters');
		const panelId = ws.activePanelId!;
		ws.linkNodeToPanel(panelId, 'osc0');
		ws.selectTab(firstTab);
		expect(ws.panelsBoundTo('osc0').map((p) => p.panelId)).toContain(panelId);
		history().reset();

		await g.removeNode('osc0');
		ws.clearNodeRefs('osc0');
		expect(ws.panelsBoundTo('osc0')).toHaveLength(0);

		await history().undo();
		expect(ws.panelsBoundTo('osc0').map((p) => p.panelId)).toContain(panelId);
	});
});

describe('deleting a collapsed sub-patch instance is undoable (manager owns the subtree capture)', () => {
	beforeEach(() => history().reset());

	it('deletes the instance via remove_node and records one undoable step', async () => {
		const { fc, g } = withInstance();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		await g.removeNode('sub');

		// Forward: the instance IS deleted via remove_node (the manager's RemoveNode captures the
		// whole subtree for its inverse — B3b).
		expect(fc.recordedCalls().some((c) => c.op === 'remove_node' && c.payload.node === 'sub')).toBe(
			true
		);
		// One undoable entry, labelled for the instance.
		expect(history().length).toBe(1);
		expect(history().undoLabel).toBe('Delete subpatch0');
		expect(history().canUndo).toBe(true);
	});

	it('undo DELEGATES to the manager (no client-side checkpoint / add_node replay)', async () => {
		const { fc, g } = withInstance();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		await g.removeNode('sub');
		const before = fc.recordedCalls().length;
		await history().undo();

		const undoCalls = fc.recordedCalls().slice(before);
		// Undo just asks the manager to undo its session command (which restores the whole subtree
		// from the inverse it captured) — never a client-side whole-patch checkpoint or a fragile
		// add_node{Sub-patch} replay.
		expect(undoCalls.some((c) => c.op === 'undo')).toBe(true);
		expect(undoCalls.some((c) => c.op === 'load')).toBe(false);
		expect(undoCalls.some((c) => c.op === 'add_node')).toBe(false);
		expect(history().canRedo).toBe(true);
	});

	it('a MIXED batch (instance + node) is ONE undo step delegating per child', async () => {
		// `sub` (holding member `m1`) plus a top-level node `n1`.
		const { fc, g } = withInstance();
		Y.transact(g.doc, () => {
			const n = new Y.Map<unknown>();
			n.set('type', 'Buffer');
			n.set('name', 'buffer1');
			nodesMap(g.doc).set('n1', n);
		});
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		await g.removeNodes(['n1', 'sub']);

		// Forward deletes both.
		const removed = fc
			.recordedCalls()
			.filter((c) => c.op === 'remove_node')
			.map((c) => c.payload.node);
		expect(removed).toContain('n1');
		expect(removed).toContain('sub');
		// ONE undoable entry (a transaction folded to a compound of two graph_cmd children).
		expect(history().length).toBe(1);
		expect(history().undoLabel).toBe('Delete 2 nodes');

		// Undo runs the compound: one manager `undo` per child (two), no client-side reload.
		const before = fc.recordedCalls().length;
		await history().undo();
		const undoCalls = fc.recordedCalls().slice(before);
		expect(undoCalls.filter((c) => c.op === 'undo')).toHaveLength(2);
		expect(undoCalls.some((c) => c.op === 'load')).toBe(false);
		expect(history().canRedo).toBe(true);

		// Redo runs the compound FORWARD: one manager `redo` per child (two), re-deleting both.
		const beforeRedo = fc.recordedCalls().length;
		await history().redo();
		const redoCalls = fc.recordedCalls().slice(beforeRedo);
		expect(redoCalls.filter((c) => c.op === 'redo')).toHaveLength(2);
		expect(history().canRedo).toBe(false);
		expect(history().canUndo).toBe(true);
	});
});
