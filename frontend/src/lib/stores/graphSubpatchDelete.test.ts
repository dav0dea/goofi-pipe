import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed, type DocSeed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import { workspace } from 'panelty';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { nodesMap } from '$lib/crdt/graphDoc';
import { SCOPE_TYPE } from '$lib/api/vocab';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';

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
		viewpoint: null
	} as never;
}

/** A store holding a collapsed sub-patch instance `sub` (one member `m1`). */
function withInstance(): { fc: FakeControl; g: GraphStore; d: DocSeed } {
	const fc = new FakeControl();
	const g = new GraphStore(fc);
	g.nodeTypes = catalog();
	// Seed as the manager sends it — the scope forest's single source.
	const d = seed(fc).patch({
		nodes: {
			m1: { type: 'Buffer', name: 'buffer0', pos: { x: 0, y: 0 }, scope: 'sub' },
			sub: { type: SCOPE_TYPE, name: 'subpatch0', pos: { x: 0, y: 0 } }
		}
	});
	return { fc, g, d };
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

// (The panels a delete empties, and their re-binding on undo, moved to the MANAGER: `RemoveNode`
// clears every panel bound to a uid it takes, so one inverse restores both. Pinned over the real
// wire by `deleting_a_node_empties_the_panels_bound_to_it_and_an_undo_binds_them_back`.)

describe('deleting a collapsed sub-patch instance is undoable (manager owns the subtree capture)', () => {
	beforeEach(() => history().reset());

	it('deletes the instance via remove_node and records one undoable step', async () => {
		const { fc, g } = withInstance();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		await g.removeNode('sub');

		// Forward: the instance IS deleted via remove_node (the manager's RemoveNode captures the
		// whole subtree for its inverse — B3b).
		expect(fc.recordedCalls().some((c) => c.op === 'node remove' && c.payload.node === 'sub')).toBe(
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
		expect(undoCalls.some((c) => c.op === 'session load')).toBe(false);
		expect(undoCalls.some((c) => c.op === 'add_node')).toBe(false);
		expect(history().canRedo).toBe(true);
	});

	it('a MIXED batch (instance + node) is ONE undo step delegating per child', async () => {
		// `sub` (holding member `m1`) plus a top-level node `n1`.
		const { fc, g, d } = withInstance();
		d.node('n1', 'Buffer', 'buffer1');
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		await g.removeNodes(['n1', 'sub']);

		// Forward deletes both.
		const removed = fc
			.recordedCalls()
			.filter((c) => c.op === 'node remove')
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
		expect(undoCalls.some((c) => c.op === 'session load')).toBe(false);
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
