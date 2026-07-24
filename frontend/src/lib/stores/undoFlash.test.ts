import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { flash } from './flash.svelte';
import { pulseRestored } from './undoFlash';
import type { NavContext } from './history.svelte';
import type { NodeTypeInfo } from '$lib/api/control';
import { nodesMap } from '$lib/crdt/graphDoc';
import * as Y from 'yjs';

/** The catalog (list_nodes) the manager provides — `g.nodeTypes = catalog()` flips the store
 * doc-authoritative, so a seeded node is built from the doc + these descriptors. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: '',
			available: true,
			dynamic: false,
			missing_deps: [],
			input_slots: {},
			output_slots: { out: 'ARRAY' },
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

describe('pulseRestored', () => {
	it('pulses the selected nodes that are on the canvas, and skips the ones that are not', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog(); // catalog present → the doc is authoritative for node identity
		// uid is the identity, name a separate mutable display label — kept DISTINCT so a
		// lookup that confuses the two is caught.
		docSeedNode(g, 'uf_present', 'Oscillator', 'display-present', [0, 0]);

		// Selection sets are keyed by uid.
		const ctx: NavContext = {
			activeWorkspaceId: 'w',
			activePanelId: 'p',
			enteredPath: {},
			selection: { p: { nodes: ['uf_present', 'uf_absent'], edges: [] } }
		};
		pulseRestored(ctx, { control: fc, graph: g, workspace: workspace() });

		expect(flash().active('uf_present')).toBe(true);
		expect(flash().active('uf_absent')).toBe(false); // not in the graph — no flash, no throw
	});
});
