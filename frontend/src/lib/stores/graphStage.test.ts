import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { nodesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo } from '$lib/api/control';
import * as Y from 'yjs';

/** The catalog (list_nodes) the manager provides — its presence flips the store to
 * doc-authoritative for node identity. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'PSD',
			category: 'signal',
			doc: '',
			available: true,
			missing_deps: [],
			input_slots: { data: 'ARRAY' },
			output_slots: { psd: 'ARRAY' },
			params: {}
		}
	];
}

/** Seed a node into the store's doc exactly as the manager's mirror (`sync_graph_to_doc`) writes it,
 * in ONE Yjs transaction so the store's afterTransaction → _syncFromDoc → reconcile fires once. */
function docSeedNode(g: GraphStore, uid: string, type: string, name: string, pos: [number, number]): void {
	Y.transact(g.doc, () => {
		const n = new Y.Map<unknown>();
		n.set('type', type);
		n.set('name', name);
		const p = new Y.Map<unknown>();
		p.set('x', pos[0]);
		p.set('y', pos[1]);
		n.set('pos', p);
		nodesMap(g.doc).set(uid, n);
	});
}

describe('node lifecycle stage', () => {
	it('seeds stage from the doc and follows state_update', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'PSD', 'psd0', [0, 0]);
		// The node's identity is doc-owned; its lifecycle stage is event-sourced, so a
		// freshly doc-seeded node carries no stage until the first state push arrives.
		expect(g.nodeById('n1')).toBeDefined();
		expect(g.nodeById('n1')?.stage).toBeUndefined();

		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, stage: 'setup' }
		});
		expect(g.nodeById('n1')?.stage).toBe('setup');

		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, stage: 'ready' }
		});
		expect(g.nodeById('n1')?.stage).toBe('ready');
	});

	it('state_update carries the error and applies it (a healthy respawn clears the stale chip)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'PSD', 'psd0', [0, 0]);

		// a setup() failure rides the idempotent state plane
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'n1',
				params: {},
				stage: 'setup',
				error: 'RuntimeError: setup boom'
			}
		});
		expect(g.nodeById('n1')?.error).toContain('setup boom');

		// a healthy respawn's state push carries error=null -> the chip clears
		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, stage: 'ready', error: null }
		});
		expect(g.nodeById('n1')?.error).toBe(null);
	});

	it('node_stage error is terminal and carries the traceback', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'PSD', 'psd0', [0, 0]);

		fc.emit({
			event: 'node_stage',
			payload: { node: 'n1', stage: 'error', error: 'ModuleNotFoundError: torch' }
		});
		expect(g.nodeById('n1')?.stage).toBe('error');
		expect(g.nodeById('n1')?.error).toContain('ModuleNotFoundError');
	});
});
