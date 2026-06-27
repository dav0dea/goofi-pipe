import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import type { NodeInstanceInfo, LinkInfo } from '$lib/api/control';

function nodeInfo(uid: string, name: string, type = 'Oscillator'): NodeInstanceInfo {
	return {
		uid,
		name,
		type,
		category: 'inputs',
		doc: '',
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		params: {},
		pos: [0, 0],
		viewers: {},
		membership: null,
		error: null
	};
}

describe('cloneNodes — identity is the uid, not the display name', () => {
	it('finds the selected nodes by uid and re-wires their internal links to the clones', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		// Display names are NOT the uids (the post-rekey reality).
		fc.emit({ event: 'node_added', payload: nodeInfo('uidA', 'oscillator0') });
		fc.emit({ event: 'node_added', payload: nodeInfo('uidB', 'buffer0', 'Buffer') });
		const link: LinkInfo = { node_out: 'uidA', slot_out: 'out', node_in: 'uidB', slot_in: 'in' };
		fc.emit({ event: 'link_added', payload: link });

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
});
