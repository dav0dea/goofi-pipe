import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { nodesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo } from '$lib/api/control';
import * as Y from 'yjs';

/** The catalog makes the doc authoritative for node + scope identity (see the node cutover). */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Buffer',
			category: 'signal',
			doc: '',
			available: true,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			params: {}
		}
	];
}

describe('root-as-scope: ROOT is synthesized as the canvas scope', () => {
	it('nodeById(ROOT_ID) is null — ROOT is the canvas, never a synth group node', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		// Any doc transaction runs the reconcile, which synthesizes the ROOT scope.
		Y.transact(g.doc, () => {
			const n = new Y.Map<unknown>();
			n.set('type', 'Buffer');
			n.set('name', 'buffer0');
			nodesMap(g.doc).set('m1', n);
		});

		expect(g.instances[ROOT_ID]).toBeDefined(); // ROOT is in the mirror
		expect(g.nodeById(ROOT_ID)).toBe(null); // …but never synthesized as a node
	});

	it('restartNode respawns in place via the restart_node RPC (keeps scope, no cascade)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);

		await g.restartNode('m1');

		const calls = fc.recordedCalls();
		expect(calls.some((c) => c.op === 'restart_node' && c.payload.node === 'm1')).toBe(true);
		// must NOT do the old remove+add dance — that lands a member back at ROOT and, for a
		// SHARED member, mirror-removes it across siblings (post Bug-C).
		expect(calls.some((c) => c.op === 'remove_node')).toBe(false);
		expect(calls.some((c) => c.op === 'add_node')).toBe(false);
	});
});
