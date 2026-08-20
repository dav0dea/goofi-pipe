import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { flash } from './flash.svelte';
import { pulseRestored } from './undoFlash';
import type { NavContext } from './history.svelte';
import type { NodeTypeInfo } from '$lib/api/control';
import { nodesMap } from '$lib/crdt/graphDoc';

/** The catalog (list_nodes) the manager provides — `g.nodeTypes = catalog()` flips the store
 * doc-authoritative, so a seeded node is built from the doc + these descriptors. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: '',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: {},
			output_slots: { out: 'ARRAY' },
			params: {}
		}
	];
}


describe('pulseRestored', () => {
	it('pulses the selected nodes that are on the canvas, and skips the ones that are not', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog(); // catalog present → the doc is authoritative for node identity
		// uid is the identity, name a separate mutable display label — kept DISTINCT so a
		// lookup that confuses the two is caught.
		d.node('uf_present', 'Oscillator', 'display-present', [0, 0]);

		// Selection sets are keyed by uid.
		const ctx: NavContext = {
			activeWorkspaceId: 'w',
			activePanelId: 'p',
			enteredPath: {},
			selection: { p: { nodes: ['uf_present', 'uf_absent'], edges: [] } }
		};
		pulseRestored(ctx, { control: fc, graph: g });

		expect(flash().active('uf_present')).toBe(true);
		expect(flash().active('uf_absent')).toBe(false); // not in the graph — no flash, no throw
	});
});
