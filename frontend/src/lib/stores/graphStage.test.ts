import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

function bootingNode(uid: string): NodeInstanceInfo {
	return {
		uid,
		name: 'psd0',
		type: 'PSD',
		category: 'signal',
		doc: '',
		input_slots: { data: 'ARRAY' },
		output_slots: { psd: 'ARRAY' },
		params: {},
		pos: [0, 0],
		viewers: {},
		membership: null,
		error: null,
		stage: 'creating'
	};
}

describe('node lifecycle stage', () => {
	it('seeds stage from node_added and follows state_update', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: bootingNode('n1') });
		expect(g.nodeById('n1')?.stage).toBe('creating');

		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, output_subscribers: {}, stage: 'setup' }
		});
		expect(g.nodeById('n1')?.stage).toBe('setup');

		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, output_subscribers: {}, stage: 'ready' }
		});
		expect(g.nodeById('n1')?.stage).toBe('ready');
	});

	it('state_update carries the error and applies it (a healthy respawn clears the stale chip)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: bootingNode('n1') });

		// a setup() failure rides the idempotent state plane
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'n1',
				params: {},
				output_subscribers: {},
				stage: 'setup',
				error: 'RuntimeError: setup boom'
			}
		});
		expect(g.nodeById('n1')?.error).toContain('setup boom');

		// a healthy respawn's state push carries error=null -> the chip clears
		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, output_subscribers: {}, stage: 'ready', error: null }
		});
		expect(g.nodeById('n1')?.error).toBe(null);
	});

	it('node_stage error is terminal and carries the traceback', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: bootingNode('n1') });

		fc.emit({
			event: 'node_stage',
			payload: { node: 'n1', stage: 'error', error: 'ModuleNotFoundError: torch' }
		});
		expect(g.nodeById('n1')?.stage).toBe('error');
		expect(g.nodeById('n1')?.error).toContain('ModuleNotFoundError');
	});
});
