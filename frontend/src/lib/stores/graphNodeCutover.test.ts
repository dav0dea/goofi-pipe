import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { nodesMap, setParamValue } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import type { ParamDescriptor } from '$lib/api/types';
import * as Y from 'yjs';

/** A minimal hello/graph_replaced snapshot; `node_types` optionally carries the palette inline. */
function helloSnap(node_types?: NodeTypeInfo[]): GraphSnapshot {
	return {
		nodes: [],
		links: [],
		instances: {},
		node_types,
		save_path: null,
		unsaved_changes: false,
		instance_id: 'sess1',
		layout: null
	} as unknown as GraphSnapshot;
}

/** The catalog (list_nodes) the manager provides — the static per-type descriptor source. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: 'A generator',
			available: true,
			dynamic: false,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			params: {
				common: {
					max_frequency: {
						type: 'float',
						value: 30,
						vmin: 0,
						vmax: 1000,
						doc: null,
						save_param: true,
						refreshable: false,
						expression: null,
						expression_enabled: false,
						expression_triggers_process: false,
						expression_error: null
					}
				}
			}
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

describe('node-identity read cutover — nodes built from the doc when the catalog is present', () => {
	it('assembles a node from doc identity + catalog descriptors', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog(); // catalog present → the doc becomes authoritative for node identity
		docSeedNode(g, 'n1', 'Oscillator', 'osc0', [10, 20]);

		const n = g.nodeById('n1');
		expect(n, 'node exists purely from the doc — no node_added event').toBeDefined();
		expect([n!.type, n!.name, n!.pos]).toEqual(['Oscillator', 'osc0', [10, 20]]);
		// Descriptors come from the catalog by type.
		expect(n!.category).toBe('inputs');
		expect(n!.output_slots).toEqual({ out: 'ARRAY' });
		expect(n!.params.common.max_frequency.type).toBe('float');
		// No doc value written → the catalog default.
		expect(n!.params.common.max_frequency.value).toBe(30);
	});

	it('reflects a doc param-value leaf-write', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'Oscillator', 'osc0', [0, 0]);

		setParamValue(g.doc, 'n1', 'common', 'max_frequency', 42); // fires a transaction → reconcile
		expect(g.nodeById('n1')!.params.common.max_frequency.value).toBe(42);
	});

	it('drops a node when it vanishes from the doc (mirror removal)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'Oscillator', 'osc0', [0, 0]);
		expect(g.nodeById('n1')).toBeDefined();

		Y.transact(g.doc, () => nodesMap(g.doc).delete('n1'));
		expect(g.nodeById('n1'), 'node removed from the doc → dropped from the store').toBeFalsy();
	});

	it('ignores node_added / node_removed events when the catalog is authoritative', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'Oscillator', 'osc0', [0, 0]);

		// A node_added for a node NOT in the doc must not mint a phantom — the reconcile owns
		// existence (else the manager's node_added would race/duplicate the doc mirror).
		fc.emit({
			event: 'node_added',
			payload: {
				uid: 'ghost',
				name: 'ghost0',
				type: 'Oscillator',
				category: 'inputs',
				doc: '',
				input_slots: {},
				output_slots: {},
				params: {},
				pos: [0, 0],
				viewers: {},
				membership: null,
				error: null
			}
		});
		expect(g.nodeById('ghost')).toBeFalsy();
		// And a node_removed for a node still IN the doc must not drop it.
		fc.emit({ event: 'node_removed', payload: { node: 'n1', membership: null } });
		expect(g.nodeById('n1')).toBeDefined();
	});

	it('state_update merges runtime bits without clobbering the doc-owned value', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docSeedNode(g, 'n1', 'Oscillator', 'osc0', [0, 0]);
		setParamValue(g.doc, 'n1', 'common', 'max_frequency', 55);

		// A state_update carrying a STALE value (999) + an expression_error + stage: the value must
		// stay the doc's 55 (params are doc-owned), while error/stage/expression_error merge.
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'n1',
				params: { common: { max_frequency: { value: 999, expression_error: 'compile error' } } } as unknown as Record<
					string,
					Record<string, ParamDescriptor>
				>,
				output_subscribers: {},
				stage: 'ready',
				error: null
			}
		});
		const n = g.nodeById('n1')!;
		expect(n.params.common.max_frequency.value).toBe(55); // doc value preserved
		expect(n.params.common.max_frequency.expression_error).toBe('compile error'); // runtime merged
		expect(n.stage).toBe('ready');
	});

	it('an unknown type (missing from the catalog) still renders identity + pos', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog(); // does NOT contain "Mystery"
		docSeedNode(g, 'n2', 'Mystery', 'mystery0', [5, 6]);

		const n = g.nodeById('n2');
		expect(n).toBeDefined();
		expect([n!.name, n!.pos]).toEqual(['mystery0', [5, 6]]);
		// Fallback: no descriptors, but the node still exists (doesn't crash the reconcile).
		expect(n!.input_slots).toEqual({});
		expect(n!.output_slots).toEqual({});
	});
});

describe('catalog-in-hello — the palette rides on the snapshot, no async list_nodes', () => {
	it('a hello carrying node_types sets the catalog synchronously (doc-authoritative from render 1)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: helloSnap(catalog()) });
		// The catalog is in hand immediately — the doc becomes authoritative with no fallback window.
		expect(g.nodeTypes?.length).toBe(1);
		// …and no async round-trip was issued for it.
		expect(fc.recordedCalls().some((c) => c.op === 'list_nodes')).toBe(false);
	});

	it('an older backend (hello without node_types) still fetches list_nodes async', () => {
		const fc = new FakeControl();
		fc.setCallResult('list_nodes', { types: catalog() });
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: helloSnap(undefined) });
		// Backward-compat: the async fetch is the fallback when the snapshot omits the palette.
		expect(fc.recordedCalls().some((c) => c.op === 'list_nodes')).toBe(true);
	});
});
