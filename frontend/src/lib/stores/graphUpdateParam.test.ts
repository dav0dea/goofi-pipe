import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed, type DocSeed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import { docParams, nodesMap, setParamValue } from '$lib/crdt/graphDoc';
import type { NodeInstanceInfo, NodeTypeInfo } from '$lib/api/control';

/** Seed a node — a param leaf-write targets the replica's node, so it no-ops unless it is there. */
function docAddNode(d: DocSeed, uid: string): void {
	d.node(uid, 'Oscillator', uid);
}

/** The catalog (list_nodes) the manager provides — `g.nodeTypes = catalog()` flips the store
 * doc-authoritative, so node identity + params come from the doc + these descriptors. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: '',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			params: {
				common: {
					frequency: {
						type: 'float',
						value: 1,
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


function nodeWithParam(uid: string, value: unknown): NodeInstanceInfo {
	return {
		uid,
		name: 'osc0',
		type: 'Oscillator',
		category: 'inputs',
		doc: '',
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		// One param group with a single param carrying `value`.
		params: { common: { frequency: { value } } } as unknown as NodeInstanceInfo['params'],
		pos: [0, 0],
		viewers: {},
		membership: null,
		error: null
	};
}

describe('GraphStore.updateParam — guards a non-existent param', () => {
	beforeEach(() => history().reset());

	it('throws (recording nothing, sending no RPC) when the param does not exist', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });

		// A missing group/name (agent typo, or a pre-hydration race) must not record a
		// poisoned undo entry whose inverse would send value:undefined → backend KeyError.
		await expect(g.updateParam('uidA', 'nope', 'missing', 5)).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'edit_node')).toBe(false);
		expect(history().canUndo).toBe(false);
	});

	it('treats a falsy current value (0) as present, not missing', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog(); // catalog present → node identity + params come from the doc
		d.node('uidA', 'Oscillator', 'osc0', [0, 0]);
		setParamValue(g.doc, 'uidA', 'common', 'frequency', 0); // the current value the guard must treat as present

		// The guard keys on the param's EXISTENCE, not the truthiness of its value, so editing a
		// param whose current value is 0/false/'' still issues the command (a missing param throws).
		await g.updateParam('uidA', 'common', 'frequency', 5);
		const call = fc.recordedCalls().find((c) => c.op === 'edit_node');
		expect(call?.payload).toEqual({ node: 'uidA', params: { common: { frequency: 5 } } });
		expect(history().canUndo).toBe(true);
	});
});

describe('GraphStore.setExpression — guards a non-existent param', () => {
	beforeEach(() => history().reset());

	it('throws (recording nothing, minting no phantom doc binding) when the param does not exist', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });
		docAddNode(d, 'uidA'); // the node exists in the doc; the PARAM does not

		// A missing group/name (agent typo, or a call racing hydration) must not leaf-write an
		// `expr` onto a phantom param entry — the graph rejects it and the re-mirror never prunes it.
		await expect(g.setExpression('uidA', 'nope', 'missing', "nd('x')")).rejects.toThrow();
		expect(history().canUndo).toBe(false);
		expect(docParams(g.doc, 'uidA').nope?.missing?.expr).toBeUndefined();
	});
});

describe('GraphStore.refreshParam — asks the node to re-evaluate options', () => {
	it('sends refresh_param with the uid/group/name and records no undo entry', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		history().reset();

		await g.refreshParam('uidA', 'audio', 'device');
		const call = fc.recordedCalls().find((c) => c.op === 'refresh_param');
		expect(call?.payload).toEqual({ node: 'uidA', group: 'audio', name: 'device' });
		// A refresh recomputes options, not values — it is not an undoable graph edit.
		expect(history().canUndo).toBe(false);
	});
});

describe('GraphStore refresh spinner — the entry stays disabled until fresh options land', () => {
	it('marks the param refreshing on refreshParam, then clears it when the node reports done', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });

		expect(g.isRefreshing('uidA', 'audio', 'device')).toBe(false);

		await g.refreshParam('uidA', 'audio', 'device');
		// The RPC only *dispatches* the ctrl message; the node re-scans asynchronously
		// (LSL resolve is ~4s), so the entry stays disabled after the RPC resolves.
		expect(g.isRefreshing('uidA', 'audio', 'device')).toBe(true);

		// The node's post-refresh state push names the completed param → spinner clears.
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'uidA',
				params: {},
				refreshed_params: [['audio', 'device']]
			}
		});
		expect(g.isRefreshing('uidA', 'audio', 'device')).toBe(false);
	});

	it('clears only the completed param, leaving other in-flight refreshes disabled', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });

		await g.refreshParam('uidA', 'audio', 'device');
		await g.refreshParam('uidA', 'lsl', 'source_name');

		// A state push completing an *unrelated* param must not lift this one's spinner.
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'uidA',
				params: {},
				refreshed_params: [['lsl', 'source_name']]
			}
		});
		expect(g.isRefreshing('uidA', 'lsl', 'source_name')).toBe(false);
		expect(g.isRefreshing('uidA', 'audio', 'device')).toBe(true);

		// Clean up the still-pending safety timeout so it can't outlive the test.
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'uidA',
				params: {},
				refreshed_params: [['audio', 'device']]
			}
		});
	});

	it('drops the spinner if the RPC dispatch itself fails (the node will never push)', async () => {
		const fc = new FakeControl();
		fc.failNext('refresh_param');
		const g = new GraphStore(fc);
		const d = seed(fc);
		fc.emit({ event: 'node_added', payload: nodeWithParam('uidA', 0) });

		await expect(g.refreshParam('uidA', 'audio', 'device')).rejects.toThrow();
		expect(g.isRefreshing('uidA', 'audio', 'device')).toBe(false);
	});
});
