import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
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
	};
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

describe('root-as-scope: ROOT ships as an instance the editor renders from', () => {
	it('nodeById(ROOT_ID) is null — ROOT is the canvas, never a synth group node', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: snapshot([], { [ROOT_ID]: instInfo(ROOT_ID, 'root', null) }) });
		expect(g.instances[ROOT_ID]).toBeDefined(); // ROOT is in the mirror
		expect(g.nodeById(ROOT_ID)).toBe(null); // …but never synthesized as a node
	});

	it('restartNode respawns in place via the restart_node RPC (keeps scope, no cascade)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		// a crashed sub-patch member
		fc.emit({
			event: 'node_added',
			payload: nodeInfo('m1', 'buffer0', { instance: 'sub', local_name: 'buffer0' })
		});

		await g.restartNode('m1');

		const calls = fc.recordedCalls();
		expect(calls.some((c) => c.op === 'restart_node' && c.payload.node === 'm1')).toBe(true);
		// must NOT do the old remove+add dance — that lands a member back at ROOT and, for a
		// SHARED member, mirror-removes it across siblings (post Bug-C).
		expect(calls.some((c) => c.op === 'remove_node')).toBe(false);
		expect(calls.some((c) => c.op === 'add_node')).toBe(false);
	});
});
