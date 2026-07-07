import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

function nodeWithParam(uid: string, value: unknown): NodeInstanceInfo {
	return {
		uid,
		name: 'osc0',
		type: 'Oscillator',
		category: 'inputs',
		doc: '',
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		// One param group with a single param carrying `value`.
		params: { common: { frequency: { value } } } as unknown as NodeInstanceInfo['params'],
		pos: [0, 0],
		viewers: {},
		membership: null,
		error: null
	};
}

describe('GraphStore.updateParam — guards a non-existent param', () => {
	beforeEach(() => history().reset());

	it('throws (recording nothing, sending no RPC) when the param does not exist', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });

		// A missing group/name (agent typo, or a pre-hydration race) must not record a
		// poisoned undo entry whose inverse would send value:undefined → backend KeyError.
		await expect(g.updateParam('uidA', 'nope', 'missing', 5)).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'update_param')).toBe(false);
		expect(history().canUndo).toBe(false);
	});

	it('treats a falsy current value (0) as present, not missing', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });

		// The guard keys on the param's existence, not the truthiness of its value, so
		// editing a param whose current value is 0/false/'' still works.
		await g.updateParam('uidA', 'common', 'frequency', 5);
		const call = fc.recordedCalls().find((c) => c.op === 'update_param');
		expect(call?.payload).toMatchObject({ node: 'uidA', group: 'common', name: 'frequency', value: 5 });
		expect(history().canUndo).toBe(true);
	});
});

describe('GraphStore.refreshParam — asks the node to re-evaluate options', () => {
	it('sends refresh_param with the uid/group/name and records no undo entry', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		history().reset();

		await g.refreshParam('uidA', 'audio', 'device');
		const call = fc.recordedCalls().find((c) => c.op === 'refresh_param');
		expect(call?.payload).toEqual({ node: 'uidA', group: 'audio', name: 'device' });
		// A refresh recomputes options, not values — it is not an undoable graph edit.
		expect(history().canUndo).toBe(false);
	});
});
