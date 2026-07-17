import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import * as Y from 'yjs';
import type { NodeInstanceInfo } from '$lib/api/control';

/** A node carrying one Float param, as node_added delivers it. */
function nodeWithParam(): NodeInstanceInfo {
	return {
		uid: 'n1',
		name: 'osc0',
		type: 'Oscillator',
		category: 'inputs',
		doc: '',
		input_slots: {},
		output_slots: { out: 'ARRAY' },
		params: {
			common: {
				max_frequency: {
					type: 'float',
					value: 10,
					vmin: 0,
					vmax: 100,
					doc: null,
					save_param: true,
					refreshable: false,
					expression: null,
					expression_enabled: false,
					expression_triggers_process: false,
					expression_error: null
				}
			}
		},
		pos: [0, 0],
		viewers: {},
		membership: null,
		error: null
	} as NodeInstanceInfo;
}

/** Write a committed param value into the store's CRDT doc — what apply_client_write mirrors
 * back after a client's direct leaf-write (no state_update is emitted for a doc-write). */
function docSetParamValue(g: GraphStore, uid: string, group: string, name: string, value: unknown): void {
	const nodes = g.doc.getMap('nodes') as Y.Map<Y.Map<unknown>>;
	let node = nodes.get(uid);
	if (!node) {
		node = new Y.Map<unknown>();
		nodes.set(uid, node);
	}
	let params = node.get('params') as Y.Map<unknown> | undefined;
	if (!params) {
		params = new Y.Map<unknown>();
		node.set('params', params);
	}
	let grp = params.get(group) as Y.Map<unknown> | undefined;
	if (!grp) {
		grp = new Y.Map<unknown>();
		params.set(group, grp);
	}
	let entry = grp.get(name) as Y.Map<unknown> | undefined;
	if (!entry) {
		entry = new Y.Map<unknown>();
		grp.set(name, entry);
	}
	entry.set('value', value);
}

describe('committed param values are read from the CRDT doc (leaf-write round-trip)', () => {
	it('a doc param-value write updates the store — the only read path for apply_client_write', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam() });
		expect(g.nodes[0].params.common.max_frequency.value).toBe(10);

		docSetParamValue(g, 'n1', 'common', 'max_frequency', 42);
		expect(g.nodes[0].params.common.max_frequency.value).toBe(42);
	});

	it('does NOT overlay an expression-enabled param (its live value must stay)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const n = nodeWithParam();
		n.params.common.max_frequency.expression_enabled = true;
		n.params.common.max_frequency.value = 7; // the live evaluated value (param_values)
		fc.emit({ event: 'node_added', payload: n });

		docSetParamValue(g, 'n1', 'common', 'max_frequency', 99); // committed literal in the doc
		expect(g.nodes[0].params.common.max_frequency.value).toBe(7); // live value preserved
	});
});
