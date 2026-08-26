import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { nodesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import type { ParamDescriptor } from '$lib/api/types';
import { slotView, isSlotExpanded } from '$lib/viewers/inlineView';

/** A minimal hello/graph_replaced snapshot; `node_types` optionally carries the palette inline. */
function helloSnap(node_types?: NodeTypeInfo[], runtime: GraphSnapshot['runtime'] = {}): GraphSnapshot {
	return {
		node_types,
		runtime,
		save_path: null,
		unsaved_changes: false,
		instance_id: 'sess1',
		viewpoint: null
	} as unknown as GraphSnapshot;
}

/** The catalog (list_nodes) the manager provides — the static per-type descriptor source. */
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
			params: {
				common: {
					max_frequency: {
						type: 'float',
						value: 30,
						vmin: 0,
						vmax: 1000,
						doc: null,
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


describe('node-identity read cutover — nodes built from the doc when the catalog is present', () => {
	it('assembles a node from doc identity + catalog descriptors', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog(); // catalog present → the doc becomes authoritative for node identity
		d.node('n1', 'Oscillator', 'osc0', [10, 20]);

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
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0]);

		d.patch({ nodes: { n1: { params: { common: { max_frequency: { value: 42 } } } } } });
		expect(g.nodeById('n1')!.params.common.max_frequency.value).toBe(42);
	});

	it('drops a node when it vanishes from the doc (mirror removal)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0]);
		expect(g.nodeById('n1')).toBeDefined();

		d.remove('nodes', 'n1');
		expect(g.nodeById('n1'), 'node removed from the doc → dropped from the store').toBeFalsy();
	});

	it('ignores a node_added announcement — the doc owns node existence', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0]);

		// node_added for a node NOT in the doc must not mint a phantom — the reconcile owns
		// existence (else the announcement would race/duplicate the doc mirror).
		fc.emit({ event: 'node_added', payload: { uid: 'ghost' } });
		expect(g.nodeById('ghost')).toBeFalsy();
		expect(g.nodeById('n1')).toBeDefined();
	});

	it('state_update merges runtime bits without clobbering the doc-owned value', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0]);
		d.patch({ nodes: { n1: { params: { common: { max_frequency: { value: 55 } } } } } });

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
		const d = seed(fc);
		g.nodeTypes = catalog(); // does NOT contain "Mystery"
		d.node('n2', 'Mystery', 'mystery0', [5, 6]);

		const n = g.nodeById('n2');
		expect(n).toBeDefined();
		expect([n!.name, n!.pos]).toEqual(['mystery0', [5, 6]]);
		// Fallback: no descriptors, but the node still exists (doesn't crash the reconcile).
		expect(n!.input_slots).toEqual({});
		expect(n!.output_slots).toEqual({});
	});
});

describe('expression live value survives a doc rebuild', () => {
	it('an expression param keeps its param_values live value across an unrelated doc rebuild', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0]);
		// Committed leaf value 99 + an ENABLED expression binding: the displayed value should track
		// the LIVE evaluation, not this committed literal.
		d.patch({
			nodes: {
				n1: {
					params: {
						common: {
							max_frequency: { value: 99, expr: { source: "nd('lfo')", enabled: true, triggers: false } }
						}
					}
				}
			}
		});

		// A param_values event delivers the live evaluated value (7) — never written to the doc.
		fc.emit({ event: 'param_values', payload: { node: 'n1', values: { common: { max_frequency: 7 } } } });
		expect(g.nodeById('n1')!.params.common.max_frequency.value).toBe(7);

		// An unrelated doc change rebuilds every node from the doc. The live value must NOT revert to
		// the committed leaf (99) — the retired fallback path guarded this; the doc path must too.
		d.node('n2', 'Oscillator', 'osc1', [1, 1]);
		expect(
			g.nodeById('n1')!.params.common.max_frequency.value,
			'live expression value preserved across the doc rebuild'
		).toBe(7);
	});
});

describe('node runtime survives a doc rebuild', () => {
	it('a survivor keeps its event-sourced stage, error and stats when an unrelated node lands', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0]);

		// The three node-level fields the doc never holds, each from its own event.
		fc.emit({ event: 'state_update', payload: { node: 'n1', params: {}, stage: 'ready', error: 'boom' } });
		fc.emit({ event: 'node_stats', payload: { node: 'n1', stats: { updates_per_second: 12.4 } } });
		expect([g.nodeById('n1')!.stage, g.nodeById('n1')!.error]).toEqual(['ready', 'boom']);

		// An unrelated doc write rebuilds every node from doc + catalog. The rebuild carries the
		// runtime forward through `_extractRuntime` → `assembleNode`, so the survivor must not be
		// blanked back to a healthy, statless boot state.
		d.node('n2', 'Oscillator', 'osc1', [1, 1]);
		const n = g.nodeById('n1')!;
		expect(n.stage, 'stage survives the rebuild').toBe('ready');
		expect(n.error, 'error survives the rebuild').toBe('boom');
		expect(n.stats, 'stats survive the rebuild').toEqual({ updates_per_second: 12.4 });
	});
});

describe('catalog-in-hello — the palette rides on the snapshot, no async list_nodes', () => {
	it('a hello carrying node_types sets the catalog synchronously (doc-authoritative from render 1)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		fc.emit({ event: 'hello', payload: helloSnap(catalog()) });
		// The catalog is in hand immediately — the doc becomes authoritative with no fallback window.
		expect(g.nodeTypes?.length).toBe(1);
		// …and no async round-trip was issued for it.
		expect(fc.recordedCalls().some((c) => c.op === 'library list')).toBe(false);
	});

	it('seeds the event-sourced runtime overlay from the snapshot', () => {
		// The runtime stream (the stats sweep) pushes only TRANSITIONS, so a client connecting to a
		// running graph would render an errored node as healthy without this seed. Structure still
		// comes from the doc alone — the snapshot carries no node list to fall back on.
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({
			event: 'hello',
			payload: helloSnap(catalog(), { n1: { stage: 'error', error: 'ImportError: no scipy' } })
		});
		// The seeder is made AFTER the hello, because that is the order the manager sends: a new
		// session's `hello` empties the replica, and the `doc_state` that follows fills it.
		seed(fc).node('n1', 'Oscillator', 'osc0', [0, 0]);

		const n = g.nodeById('n1')!;
		expect(n.error).toBe('ImportError: no scipy');
		expect(n.stage).toBe('error');
	});

	it('applies the runtime overlay to nodes the doc delta already materialized', () => {
		// A load sends the CRDT delta and the `graph_replaced` JSON on two channels of ONE socket,
		// and the bridge's `select!` picks between two ready branches at random — so the delta can
		// arrive FIRST, materializing every freshly minted uid before the overlay is in hand. The
		// snapshot is authoritative for runtime by definition, so it has to apply to nodes that
		// already exist, not only seed ones that materialize afterwards.
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: helloSnap(catalog()) });
		seed(fc).node('n9', 'Oscillator', 'osc0', [0, 0]);
		expect(g.nodeById('n9')!.error).toBeFalsy();

		fc.emit({
			event: 'graph_replaced',
			payload: helloSnap(catalog(), { n9: { stage: 'error', error: 'ImportError: no scipy' } })
		});

		const n = g.nodeById('n9')!;
		expect(n.stage, 'the overlay applies in the delta-first order too').toBe('error');
		expect(n.error).toBe('ImportError: no scipy');
	});
});

describe('inline viewer state — the document is the one holder, so it follows the document', () => {
	/** The blob as the projection writes it: a JSON STRING leaf under the node. */
	const blob = (v: Record<string, unknown>) => ({ viewers: JSON.stringify(v) });

	it('a peer’s collapse and kind land in this tab too', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'Oscillator', 'osc0', [0, 0], blob({ out: { collapsed: true, kind: 'image' } }));
		expect(isSlotExpanded(g.nodeById('n1'), 'out'), 'the saved collapse is what we draw').toBe(false);
		expect(slotView(g.nodeById('n1'), 'out').kind).toBe('image');

		// A SECOND tab expands the slot and picks another viewer. It reaches us as an ordinary doc
		// patch on a node we already hold — the case a client-side copy, seeded once per new uid,
		// never hears about: the two tabs then disagree, and whichever edits next overwrites the peer.
		d.patch({ nodes: { n1: blob({ out: { collapsed: false, kind: 'topomap' } }) } });
		expect(isSlotExpanded(g.nodeById('n1'), 'out'), 'the peer’s expand converges').toBe(true);
		expect(slotView(g.nodeById('n1'), 'out').kind, 'and so does the peer’s kind').toBe('topomap');
	});

	it('an in-session load onto a re-used uid draws the loaded patch, not the outgoing one', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = [...catalog(), { ...catalog()[0], type: 'Buffer' }];
		fc.emit({ event: 'hello', payload: helloSnap() });
		const d = seed(fc);
		d.node('n1', 'Oscillator', 'osc0', [0, 0], blob({ out: { collapsed: true, kind: 'image' } }));
		expect(isSlotExpanded(g.nodeById('n1'), 'out')).toBe(false);

		// A patch is loaded INTO the running session: same backend (so no generation reset), and it
		// restores the uids it was saved with — so `n1` is a SURVIVOR carrying a different node.
		fc.emit({ event: 'graph_replaced', payload: helloSnap() });
		d.push({ nodes: { n1: { type: 'Buffer', name: 'buf0', pos: { x: 0, y: 0 } } }, links: [], instances: {}, globals: {}, arrangement: {} });

		expect(g.nodeById('n1')!.type, 'precondition: the uid now carries the loaded node').toBe('Buffer');
		expect(isSlotExpanded(g.nodeById('n1'), 'out'), 'the outgoing collapse did not stick').toBe(true);
		expect(slotView(g.nodeById('n1'), 'out').kind, 'nor the outgoing kind').toBeUndefined();
	});

	it('a write carries the whole blob, touches one slot, and invents no kind', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = [{ ...catalog()[0], output_slots: { out: 'ARRAY', sig: 'ARRAY' } }];
		// `gone` is a slot the saved blob names and the node no longer has — a node file that lost an
		// output since the patch was written. It must not block an edit to a slot that does exist.
		d.node('n1', 'Oscillator', 'osc0', [0, 0], blob({ out: { kind: 'image' }, sig: { collapsed: true }, gone: { collapsed: true } }));

		g.setSlotView('n1', 'sig', { collapsed: false });
		const call = fc.recordedCalls().find((c) => c.op === 'node edit');
		expect(call!.payload, 'only the slot edited travels — the manager merges the rest').toEqual({
			node: 'n1',
			viewer: [{ slot: 'sig', collapsed: false }]
		});
		// Nothing changed until the manager answers — the document is the holder, not this store.
		expect(isSlotExpanded(g.nodeById('n1'), 'sig')).toBe(false);
		d.patch({
			nodes: { n1: blob({ out: { kind: 'image' }, sig: { collapsed: false }, gone: { collapsed: true } }) }
		});
		expect(isSlotExpanded(g.nodeById('n1'), 'sig')).toBe(true);
	});
});
