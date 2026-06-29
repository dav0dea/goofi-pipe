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
		viewers: {},
		member_count: Object.keys(members).length
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

	it('a top-level node_added is folded into ROOT.members (so it renders as a root child)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: snapshot([], { [ROOT_ID]: instInfo(ROOT_ID, 'root', null) }) });

		fc.emit({
			event: 'node_added',
			payload: nodeInfo('n1', 'buffer0', { instance: ROOT_ID, local_name: 'buffer0' })
		});

		expect(g.instances[ROOT_ID].members['buffer0']).toEqual({ uid: 'n1', is_instance: false });
		expect(g.instances[ROOT_ID].member_count).toBe(1);
	});

	it('a member node_added is folded into the owning sub-patch instance', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const root = instInfo(ROOT_ID, 'root', null, { subpatch0: { uid: 'sub', is_instance: true } });
		const sub = instInfo('sub', 'subpatch0', ROOT_ID, {});
		fc.emit({ event: 'hello', payload: snapshot([], { [ROOT_ID]: root, sub }) });

		fc.emit({
			event: 'node_added',
			payload: nodeInfo('m1', 'buffer0', { instance: 'sub', local_name: 'buffer0' })
		});

		expect(g.instances['sub'].members['buffer0']).toEqual({ uid: 'm1', is_instance: false });
		expect(g.instances['sub'].member_count).toBe(1);
	});

	it('a member node_removed is dropped from the owning instance members map', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const root = instInfo(ROOT_ID, 'root', null, { subpatch0: { uid: 'sub', is_instance: true } });
		const sub = instInfo('sub', 'subpatch0', ROOT_ID, { buffer0: { uid: 'm1', is_instance: false } });
		fc.emit({ event: 'hello', payload: snapshot([nodeInfo('m1', 'buffer0', { instance: 'sub', local_name: 'buffer0' })], { [ROOT_ID]: root, sub }) });
		expect(g.instances['sub'].member_count).toBe(1);

		fc.emit({
			event: 'node_removed',
			payload: { node: 'm1', membership: { instance: 'sub', local_name: 'buffer0' } }
		});

		expect(g.instances['sub'].members['buffer0']).toBeUndefined();
		expect(g.instances['sub'].member_count).toBe(0);
	});

	it('a top-level node_removed is dropped from ROOT.members', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const root = instInfo(ROOT_ID, 'root', null, { buffer0: { uid: 'n1', is_instance: false } });
		fc.emit({ event: 'hello', payload: snapshot([nodeInfo('n1', 'buffer0', { instance: ROOT_ID, local_name: 'buffer0' })], { [ROOT_ID]: root }) });

		fc.emit({
			event: 'node_removed',
			payload: { node: 'n1', membership: { instance: ROOT_ID, local_name: 'buffer0' } }
		});

		expect(g.instances[ROOT_ID].members['buffer0']).toBeUndefined();
		expect(g.instances[ROOT_ID].member_count).toBe(0);
	});
});
