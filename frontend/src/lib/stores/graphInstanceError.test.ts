import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { nodesMap, instancesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';

/**
 * A collapsed sub-patch's health is DERIVED from its descendants and cached in
 * `instances[id].error`. Only the discrete `error` event recomputed that cache — but a node's
 * `error` field is written by three other paths, and the most important of them is the one
 * designed to heal a missed transition:
 *
 *   - `state_update` — the DURABLE plane, re-pushed, which the code's own comment describes as
 *     what surfaces a lost first PROCESSING_ERROR after a reconnect;
 *   - `node_stage` — the terminal bootstrap error (an import failure, no auto-restart);
 *   - `_replaceSnapshot`'s runtime overlay — what a reconnecting client is handed.
 *
 * So exactly on the recovery paths, a collapsed sub-patch drew healthy while a member inside it
 * was failing. Each test below drives ONE of those paths with no `error` event anywhere near it.
 */
describe('GraphStore — a collapsed sub-patch follows every path that writes a node error', () => {
	it('updates and recovers on `state_update` alone', () => {
		const { g, fc } = subpatchWithOneMember();
		expect(g.instances['i1'].error).toBeFalsy();

		fc.emit({ event: 'state_update', payload: { node: 'n1', error: 'boom', params: {} } });
		expect(g.instances['i1'].error, 'the facade shows its member failing').toBe('boom');

		fc.emit({ event: 'state_update', payload: { node: 'n1', error: null, params: {} } });
		expect(g.instances['i1'].error, 'and recovers with it').toBeFalsy();
	});

	it('updates on `node_stage` alone', () => {
		const { g, fc } = subpatchWithOneMember();
		fc.emit({
			event: 'node_stage',
			payload: { node: 'n1', stage: 'error', error: 'ImportError: no scipy' }
		});
		expect(g.instances['i1'].error).toBe('ImportError: no scipy');
	});

	it('updates from a reconnect snapshot runtime overlay', () => {
		const { g, fc } = subpatchWithOneMember();
		// Same instance_id: a transient reconnect, not a new session — the replica survives and no
		// doc transaction fires, so nothing else would recompute the derived cache.
		fc.emit({
			event: 'hello',
			payload: snap('sess1', { n1: { stage: 'error', error: 'crashed while we were away' } })
		});
		expect(g.instances['i1'].error).toBe('crashed while we were away');
	});
});

/** A sub-patch instance `i1` holding one member node `n1`, seeded as the manager's mirror writes it. */
function subpatchWithOneMember(): { g: GraphStore; fc: FakeControl } {
	const fc = new FakeControl();
	const g = new GraphStore(fc);
	g.nodeTypes = catalog();
	fc.emit({ event: 'hello', payload: snap('sess1') });
	seed(fc).patch({
		nodes: { n1: { type: 'Oscillator', name: 'osc0', pos: { x: 0, y: 0 } } },
		instances: {
			i1: {
				name: 'sub0',
				parent: 'ROOT',
				members: { n1: { is_instance: false } },
				pos: { x: 0, y: 0 }
			}
		}
	});
	return { g, fc };
}

function snap(instance_id: string, runtime: GraphSnapshot['runtime'] = {}): GraphSnapshot {
	return {
		runtime,
		save_path: null,
		unsaved_changes: false,
		instance_id,
		viewpoint: null
	} as unknown as GraphSnapshot;
}

function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: 'A generator',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			params: {}
		} as unknown as NodeTypeInfo
	];
}
