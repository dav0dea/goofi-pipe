import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { NodeInstanceInfo, InstanceInfo, GraphSnapshot } from '$lib/api/control';

function nodeInfo(
	uid: string,
	name: string,
	membership: { instance: string; local_name: string } | null = null
): NodeInstanceInfo {
	return {
		uid,
		name,
		type: 'Buffer',
		category: 'signal',
		doc: '',
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		params: {},
		pos: [0, 0],
		viewers: {},
		membership,
		error: null
	} as NodeInstanceInfo;
}

function instInfo(
	uid: string,
	name: string,
	parent: string | null,
	members: Record<string, { uid: string; is_instance: boolean }> = {}
): InstanceInfo {
	return {
		uid,
		name,
		kind: 'subpatch',
		def_id: null,
		parent,
		interface: {},
		pos: [0, 0],
		members,
		slots: { input: {}, output: {} },
		siblings: [],
		error: null,
		viewers: {}
	} as InstanceInfo;
}

function snapshot(
	nodes: NodeInstanceInfo[],
	instances: Record<string, InstanceInfo>
): GraphSnapshot {
	return {
		nodes,
		links: [],
		instances,
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
	const root = instInfo(ROOT_ID, 'root', null, { subpatch0: { uid: 'sub', is_instance: true } });
	const sub = instInfo('sub', 'subpatch0', ROOT_ID, { buffer0: { uid: 'm1', is_instance: false } });
	fc.emit({
		event: 'hello',
		payload: snapshot([nodeInfo('m1', 'buffer0', { instance: 'sub', local_name: 'buffer0' })], {
			[ROOT_ID]: root,
			sub
		})
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
		fc.emit({ event: 'hello', payload: snapshot([], {}) });
		history().record({ kind: 'graph_cmd', domain: 'graph', label: 'Add X', context: ctx });
		expect(history().canUndo).toBe(true);

		// A load replaces the graph in the SAME backend session — the manager cleared its command
		// history (load_text → CommandHistory::clear), so the client's stale entries would pop
		// mismatched. graph_replaced must reset the client history too.
		fc.emit({ event: 'graph_replaced', payload: snapshot([], {}) });
		expect(history().canUndo).toBe(false);
		void g;
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
		// from the inverse it captured) — never a client-side load_text checkpoint or a fragile
		// add_node{Sub-patch} replay.
		expect(undoCalls.some((c) => c.op === 'undo')).toBe(true);
		expect(undoCalls.some((c) => c.op === 'load_text')).toBe(false);
		expect(undoCalls.some((c) => c.op === 'add_node')).toBe(false);
		expect(history().canRedo).toBe(true);
	});

	it('a MIXED batch (instance + node) is ONE undo step delegating per child', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const root = instInfo(ROOT_ID, 'root', null, {
			subpatch0: { uid: 'sub', is_instance: true },
			buffer1: { uid: 'n1', is_instance: false }
		});
		const sub = instInfo('sub', 'subpatch0', ROOT_ID, { buffer0: { uid: 'm1', is_instance: false } });
		fc.emit({
			event: 'hello',
			payload: snapshot(
				[
					nodeInfo('n1', 'buffer1', { instance: ROOT_ID, local_name: 'buffer1' }),
					nodeInfo('m1', 'buffer0', { instance: 'sub', local_name: 'buffer0' })
				],
				{ [ROOT_ID]: root, sub }
			)
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
		expect(undoCalls.some((c) => c.op === 'load_text')).toBe(false);
		expect(history().canRedo).toBe(true);
	});
});
