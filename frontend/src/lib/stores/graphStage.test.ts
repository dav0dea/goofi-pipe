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

	it('node_stage error clears a lingering crashed flag (boot failure after a crash)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: bootingNode('n1') });
		fc.emit({ event: 'node_crashed', payload: { node: 'n1', exitcode: -9, restarts: 1 } });
		expect(g.nodeById('n1')?.crashed).toBe(true);

		// the respawn dies during import before its first state push -> terminal error.
		fc.emit({ event: 'node_stage', payload: { node: 'n1', stage: 'error', error: 'boot tb' } });
		expect(g.nodeById('n1')?.crashed).toBe(false); // not stuck 'restarting' forever
		expect(g.nodeById('n1')?.error).toContain('boot tb');
	});

	it('a stale subpatch_changed snapshot does not regress a ready node to booting', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: bootingNode('n1') });
		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, output_subscribers: {}, stage: 'ready' }
		});
		expect(g.nodeById('n1')?.stage).toBe('ready');

		// a snapshot built on a manager thread BEFORE the node reached ready arrives late.
		fc.emit({
			event: 'subpatch_changed',
			payload: {
				instance_id: 'x',
				nodes: [{ ...bootingNode('n1'), stage: 'creating' }],
				links: [],
				instances: {},
				save_path: null,
				unsaved_changes: false,
				layout: null
			}
		});
		expect(g.nodeById('n1')?.stage).toBe('ready'); // stale pre-ready stage ignored
	});

	it('a stale subpatch_changed snapshot does not regress a survivor runtime state (error/stats/restarts)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: bootingNode('n1') });
		// the node settles via the authoritative state stream: ready, error cleared, live stats.
		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, output_subscribers: {}, stage: 'ready', error: null }
		});
		fc.emit({ event: 'node_stats', payload: { node: 'n1', stats: { updates_per_second: 30, mean_process_ms: 1, total_ticks: 100 } } });
		fc.emit({ event: 'node_crashed', payload: { node: 'n1', exitcode: -9, restarts: 2 } });
		expect(g.nodeById('n1')?.stats).toEqual({ updates_per_second: 30, mean_process_ms: 1, total_ticks: 100 });
		expect(g.nodeById('n1')?.restarts).toBe(2);

		// a structure snapshot built BEFORE the node settled carries stale runtime state.
		fc.emit({
			event: 'subpatch_changed',
			payload: {
				instance_id: 'x',
				nodes: [
					{
						...bootingNode('n1'),
						stage: 'creating',
						error: 'stale setup boom',
						stats: null,
						restarts: 0,
						log_endpoint: null
					}
				],
				links: [],
				instances: {},
				save_path: null,
				unsaved_changes: false,
				layout: null
			}
		});
		// runtime lifecycle is owned by the state stream, not the structure event.
		expect(g.nodeById('n1')?.stage).toBe('ready');
		expect(g.nodeById('n1')?.error).toBe(null); // stale error not resurrected
		expect(g.nodeById('n1')?.stats).toEqual({ updates_per_second: 30, mean_process_ms: 1, total_ticks: 100 }); // live stats kept
		expect(g.nodeById('n1')?.restarts).toBe(2); // live restart count kept
	});
});
