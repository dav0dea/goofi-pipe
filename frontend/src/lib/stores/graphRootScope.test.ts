import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { NodeTypeInfo } from '$lib/api/control';

/** The catalog makes the doc authoritative for node + scope identity (see the node cutover). */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Buffer',
			category: 'signal',
			doc: '',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: { in: 'ARRAY' },
			output_slots: { out: 'ARRAY' },
			params: {}
		}
	];
}

describe('root-as-scope: ROOT is the canvas, and no record is it', () => {
	it('a top-level record NAMES the root scope, and ROOT itself resolves to no node', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		seed(fc).node('m1', 'Buffer', 'buffer0');

		expect(g.nodeById('m1')!.scope).toBe(ROOT_ID); // a record with no `scope` is drawn at ROOT
		expect(g.nodeById(ROOT_ID)).toBe(null); // …and the sentinel names no record
	});

	it('restartNode respawns in place via the restart_node RPC (keeps scope, no cascade)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);

		await g.restartNode('m1');

		const calls = fc.recordedCalls();
		expect(calls.some((c) => c.op === 'restart_node' && c.payload.node === 'm1')).toBe(true);
		// must NOT do the old remove+add dance — that lands a member back at ROOT and, for a
		// SHARED member, mirror-removes it across siblings (post Bug-C).
		expect(calls.some((c) => c.op === 'remove_node')).toBe(false);
		expect(calls.some((c) => c.op === 'add_node')).toBe(false);
	});
});
