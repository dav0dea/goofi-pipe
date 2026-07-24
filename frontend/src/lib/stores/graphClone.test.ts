import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import type { LinkInfo, NodeTypeInfo } from '$lib/api/control';
import * as Y from 'yjs';
import { linksArray, nodesMap, setParamValue } from '$lib/crdt/graphDoc';

/** Links are read from the CRDT doc (Phase 2); seed one by writing the doc. */
function docAddLink(g: GraphStore, link: LinkInfo): void {
	const m = new Y.Map<unknown>();
	m.set('node_out', link.node_out);
	m.set('slot_out', link.slot_out);
	m.set('node_in', link.node_in);
	m.set('slot_in', link.slot_in);
	linksArray(g.doc).push([m]);
}

/** The catalog (list_nodes) the manager provides — `g.nodeTypes = catalog()` flips the store
 * doc-authoritative for node identity, so nodes are built from the doc + these descriptors. */
function catalog(): NodeTypeInfo[] {
	const floatParam = (value: number) => ({
		type: 'float' as const,
		value,
		vmin: 0,
		vmax: 1000,
		doc: null,
		refreshable: false,
		expression: null,
		expression_enabled: false,
		expression_triggers_process: false,
		expression_error: null
	});
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: '',
			available: true,
			dynamic: false,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			// Default 30 — so a seeded 42 is a genuine non-default the clone must carry inline.
			params: { common: { max_frequency: floatParam(30) } }
		},
		{
			type: 'Buffer',
			category: 'inputs',
			doc: '',
			available: true,
			dynamic: false,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
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

describe('cloneNodes — identity is the uid, not the display name', () => {
	it('finds the selected nodes by uid and re-wires their internal links to the clones', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog(); // catalog present → the doc is authoritative for node identity
		// Display names are NOT the uids (the post-rekey reality).
		docSeedNode(g, 'uidA', 'Oscillator', 'oscillator0', [0, 0]);
		docSeedNode(g, 'uidB', 'Buffer', 'buffer0', [0, 0]);
		const link: LinkInfo = { node_out: 'uidA', slot_out: 'out', node_in: 'uidB', slot_in: 'in' };
		docAddLink(g, link);

		fc.setCallResult('add_node', 'NEW'); // each clone resolves to this new uid

		// The only callers (NodeEditorPanel.selectedNodeNames / agent surface) pass UIDS.
		const rename = await g.cloneNodes(['uidA', 'uidB']);

		// The selection was found by uid (a name-keyed filter returns {} → no clone).
		expect(Object.keys(rename).sort()).toEqual(['uidA', 'uidB']);

		// The internal link is remapped onto the clones — not left pointing at the
		// originals (which a name-keyed rename map would do, since endpoints are uids).
		const addLink = fc.recordedCalls().find((c) => c.op === 'add_link');
		expect(addLink, 'cloneNodes must re-create the internal link').toBeDefined();
		expect(addLink!.payload).toMatchObject({ node_out: 'NEW', node_in: 'NEW' });
	});

	it('carries cloned param values INLINE on add_node — no racy post-add leaf-write', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'uidA', 'Oscillator', 'osc0', [0, 0]);
		// A non-default param value the clone must carry (default is 30).
		setParamValue(g.doc, 'uidA', 'common', 'max_frequency', 42);
		fc.setCallResult('add_node', 'NEW');

		await g.cloneNodes(['uidA']);

		// The value rides inline on add_node (applied under the graph lock, node born configured) —
		// NOT written via a post-add leaf-write that would no-op until the new node syncs into the
		// replica (which it never does in this test, so a missed inline would drop the value).
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node');
		expect(addCall?.payload).toMatchObject({ type: 'Oscillator', params: { common: { max_frequency: 42 } } });
	});
});
