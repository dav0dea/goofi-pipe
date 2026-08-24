import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';

/**
 * Copy, paste and duplicate are ONE pair of manager ops. What the store owes them is the uids the
 * user selected, the fragment verbatim, and the rename map back — never a re-derived spec, because
 * a spec is a per-TYPE description and a sub-patch is not one type.
 */
describe('copy / paste / duplicate — the store carries the manager’s fragment, and nothing else', () => {
	it('duplicates by copying and pasting, so a sub-patch goes through the leaf’s door', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		d.node('uidA', 'Oscillator', 'oscillator0', [0, 0]);
		d.instance('i1', 'subpatch0', [10, 10]);
		const doc = { nodes: { uidA: { pos: [0, 0] }, i1: { pos: [10, 10] } }, links: [] };
		fc.setCallResult('copy_nodes', { doc });
		fc.setCallResult('paste_nodes', { rename: { uidA: 'newA', i1: 'newI' } });

		const rename = await g.cloneNodes(['uidA', 'i1']);
		expect(rename).toEqual({ uidA: 'newA', i1: 'newI' });

		const copy = fc.recordedCalls().find((c) => c.op === 'copy_nodes');
		expect(copy?.payload, 'the selection goes as uids — a facade among them').toEqual({
			nodes: ['uidA', 'i1']
		});
		const paste = fc.recordedCalls().find((c) => c.op === 'paste_nodes');
		expect(paste?.payload.doc, 'the fragment is handed back VERBATIM').toEqual(doc);
		expect(paste?.payload.pos, 'a duplicate lands beside its original').toEqual([40, 40]);
	});

	it('pastes into the entered sub-patch, at the shift the caller asked for', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seed(fc);
		fc.setCallResult('paste_nodes', { rename: {} });

		await g.pasteNodes({ nodes: { a: { pos: [0, 0] } } }, [7, 9], 'i1');
		const paste = fc.recordedCalls().find((c) => c.op === 'paste_nodes');
		expect(paste?.payload).toMatchObject({ pos: [7, 9], inst_id: 'i1' });

		// …and at root, `inst_id` is explicitly null rather than absent: the manager reads a missing
		// key and a null one the same way, and saying it is what keeps the two ends in step.
		await g.pasteNodes({ nodes: {} });
		const rooted = fc.recordedCalls().filter((c) => c.op === 'paste_nodes').at(-1);
		expect(rooted?.payload.inst_id).toBeNull();
	});

	it('copies nothing when nothing is selected, rather than pasting an empty fragment', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seed(fc);
		expect(await g.cloneNodes([])).toEqual({});
		expect(fc.recordedCalls().some((c) => c.op === 'copy_nodes' || c.op === 'paste_nodes')).toBe(false);
	});
});
