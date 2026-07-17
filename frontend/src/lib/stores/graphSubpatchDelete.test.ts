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

describe('deleting a collapsed sub-patch instance is undoable (no data loss)', () => {
	beforeEach(() => history().reset());

	it('records a load_patch checkpoint, not a remove_node whose undo replays add_node{Sub-patch}', async () => {
		const { fc, g } = withInstance();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.setCallResult('serialize', { yaml: 'PATCH_WITH_SUBPATCH' });

		await g.removeNode('sub');

		// Forward: the instance IS actually deleted via remove_node (routes to remove_instance).
		expect(fc.recordedCalls().some((c) => c.op === 'remove_node' && c.payload.node === 'sub')).toBe(
			true
		);
		// One undoable entry, labelled for the instance.
		expect(history().length).toBe(1);
		expect(history().undoLabel).toBe('Delete subpatch0');
		expect(history().canUndo).toBe(true);
	});

	it('undo restores the subtree via a full-graph checkpoint (never a broken add_node)', async () => {
		const { fc, g } = withInstance();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.setCallResult('serialize', { yaml: 'PATCH_WITH_SUBPATCH' });

		await g.removeNode('sub');
		await history().undo();

		const calls = fc.recordedCalls();
		// Undo replays the checkpoint (load_text with the pre-delete YAML) — reconstructing the
		// entire subtree — and NEVER the uninstantiable add_node{type:'Sub-patch'} that lost data.
		const load = calls.find((c) => c.op === 'load_text');
		expect(load, 'undo restores via load_text checkpoint').toBeTruthy();
		expect(load!.payload.content).toBe('PATCH_WITH_SUBPATCH');
		expect(
			calls.some((c) => c.op === 'add_node' && c.payload.type === 'Sub-patch'),
			'must never replay the uninstantiable Sub-patch add_node'
		).toBe(false);
		expect(history().canRedo).toBe(true);
	});
});
