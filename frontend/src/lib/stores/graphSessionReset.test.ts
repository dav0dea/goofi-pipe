import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed, type DocSeed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { rawInlineView, setInlineKind } from '$lib/viewers/inlineView.svelte';
import { ui } from './ui.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';

/**
 * A fresh backend session is a GENERATION boundary, not merely a document swap.
 *
 * `SyncClient.reset()` correctly hands the store an empty replica — but the Svelte projections
 * assembled from the OLD document (`nodes`, `links`, `instances`, `globals`, and the per-uid
 * inline-view / slot-expansion stores) are plain state that nothing clears. Reconciliation runs
 * only from the doc observer, and only when `txn.changed.size > 0`.
 *
 * That is the trap: a fresh manager whose graph is EMPTY produces a sync transaction that changes
 * no Yjs type, so the observer never fires and the browser keeps rendering the graph that went
 * away — on uids the new session is about to mint again from 1.
 *
 * So the fixture below must deliver an EMPTY transaction. A fixture that seeds the replacement
 * document with content would reconcile through the ordinary path and pass against the bug.
 */
describe('GraphStore — a new backend session clears what the old one drew', () => {
	it('clears every document-derived projection, even when the fresh doc changes no type', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		fc.emit({ event: 'hello', payload: snap('sess1') });

		hydrate(fc);
		expect(g.nodes.length, 'precondition: the old session is on screen').toBe(2);
		expect(g.links.length).toBe(1);
		// Set it FALSE: `isSlotExpanded` defaults to true when the entry is absent, so a `true`
		// fixture could not tell "cleared" from "still set".
		ui().setSlotExpanded('n1', 'out', false);
		expect(ui().isSlotExpanded('n1', 'out')).toBe(false);
		setInlineKind('n1', 'out', 'image');
		// `rawInlineView` FALLS BACK to an empty view rather than returning undefined, so asserting
		// on its presence would hold against any implementation. Assert on the kind it carries.
		expect(rawInlineView('n1', 'out').kind).toBe('image');

		// A NEW backend session. The store resets the replica, and the fresh session's `doc_state`
		// carries an empty document — which is the case a reset has to survive, because nothing in
		// the arriving document mentions the old session's uids at all.
		fc.emit({ event: 'hello', payload: snap('sess2') });
		seed(fc);

		expect(g.nodes, 'nodes').toEqual([]);
		expect(g.links, 'links').toEqual([]);
		expect(g.globals, 'globals').toEqual([]);
		// ROOT is SYNTHESIZED on every reconcile — it is the canvas, not a scope the manager sent —
		// so the fresh session has one, holding nothing. Asserting `{}` here would only pass against
		// a fixture that reconciled nothing at all, which is not what a new session does.
		expect(Object.keys(g.instances), 'instances').toEqual([ROOT_ID]);
		expect(g.instances[ROOT_ID].members, 'and it kept no member of the old session').toEqual({});
		expect(rawInlineView('n1', 'out').kind, 'per-uid inline view state').toBeUndefined();
		expect(ui().isSlotExpanded('n1', 'out'), 'per-uid slot expansion is back to its default').toBe(
			true
		);
	});

	it('leaves a same-session reconnect alone', () => {
		// The other half: a transient drop re-delivers `hello` with the SAME instance_id, and the
		// replica is still valid. Clearing there would blank the canvas on every reconnect.
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		fc.emit({ event: 'hello', payload: snap('sess1') });
		hydrate(fc);

		fc.emit({ event: 'hello', payload: snap('sess1') });

		expect(g.nodes.length, 'the graph survives a reconnect to the same backend').toBe(2);
		expect(g.links.length).toBe(1);
	});
});

/** Two nodes and a link, the way the manager sends them. */
function hydrate(fc: FakeControl): DocSeed {
	return seed(fc).patch({
		nodes: {
			n1: { type: 'Oscillator', name: 'osc-n1', pos: { x: 0, y: 0 } },
			n2: { type: 'Oscillator', name: 'osc-n2', pos: { x: 0, y: 0 } }
		},
		links: [{ node_out: 'n1', slot_out: 'out', node_in: 'n2', slot_in: 'in' }]
	});
}

function snap(instance_id: string): GraphSnapshot {
	return {
		runtime: {},
		save_path: null,
		unsaved_changes: false,
		instance_id,
		viewpoint: null
	} as unknown as GraphSnapshot;
}

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
			params: {}
		} as unknown as NodeTypeInfo
	];
}
