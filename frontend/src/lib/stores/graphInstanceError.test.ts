import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { nodesMap, instancesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import * as Y from 'yjs';

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
	Y.transact(g.doc, () => {
		const n = new Y.Map<unknown>();
		n.set('type', 'Oscillator');
		n.set('name', 'osc0');
		const p = new Y.Map<unknown>();
		p.set('x', 0);
		p.set('y', 0);
		n.set('pos', p);
		nodesMap(g.doc).set('n1', n);

		const inst = new Y.Map<unknown>();
		inst.set('name', 'sub0');
		inst.set('parent', 'ROOT');
		const members = new Y.Map<unknown>();
		const member = new Y.Map<unknown>();
		member.set('is_instance', false);
		members.set('n1', member);
		inst.set('members', members);
		const ipos = new Y.Map<unknown>();
		ipos.set('x', 0);
		ipos.set('y', 0);
		inst.set('pos', ipos);
		instancesMap(g.doc).set('i1', inst);
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
