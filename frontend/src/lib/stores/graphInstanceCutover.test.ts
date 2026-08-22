import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { nodesMap } from '$lib/crdt/graphDoc';
import { SCOPE_TYPE, BOUNDARY_SLOT, boundaryType } from '$lib/api/vocab';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import { slotView, isSlotExpanded } from '$lib/viewers/inlineView';

/** Minimal catalog — its presence flips the store to doc-authoritative identity. */
function catalog(): NodeTypeInfo[] {
	const mk = (type: string): NodeTypeInfo => ({
		type,
		category: 'array',
		doc: '',
		source: 'builtin',
		available: true,
		missing_deps: [],
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		params: {}
	});
	return [mk('Oscillator'), mk('Buffer')];
}

/** A node, as the projection writes it. `scope` names the sub-patch it belongs to; a top-level
 * record simply omits the key. */
const node = (type: string, name: string, scope?: string) => ({
	type,
	name,
	pos: { x: 0, y: 0 },
	...(scope ? { scope } : {})
});

interface Port {
	uid: string;
	type: string;
	name: string;
	pos?: [number, number];
	/** `[member uid, slot]` the port's inner wire names, or nothing when it is unwired. */
	inner?: [string, string];
}

/** A scope in the exact shape the projection writes it: a facade record, one record per port, and
 * one LINK per inner wire. Membership rides each member's own record, so seed those with
 * `node(type, name, uid)`. */
function scope(
	uid: string,
	o: { name: string; parent?: string; pos?: [number, number]; ports?: Port[] }
): { nodes: Record<string, unknown>; links: unknown[] } {
	const pos = o.pos ?? [0, 0];
	const nodes: Record<string, unknown> = {
		[uid]: node(SCOPE_TYPE, o.name, o.parent === ROOT_ID ? undefined : o.parent)
	};
	(nodes[uid] as Record<string, unknown>).pos = { x: pos[0], y: pos[1] };
	const links: unknown[] = [];
	for (const p of o.ports ?? []) {
		const pp = p.pos ?? [0, 0];
		nodes[p.uid] = { type: p.type, name: p.name, pos: { x: pp[0], y: pp[1] }, scope: uid };
		if (!p.inner) continue;
		const [n, s] = p.inner;
		links.push(
			boundaryType(p.type)!.dir === 'in'
				? { node_out: p.uid, slot_out: BOUNDARY_SLOT, node_in: n, slot_in: s }
				: { node_out: n, slot_out: s, node_in: p.uid, slot_in: BOUNDARY_SLOT }
		);
	}
	return { nodes, links };
}

describe('scope-forest read cutover — scopes built from the doc when the catalog is present', () => {
	it('synthesizes ROOT and assembles a scope from the doc forest', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const sp = scope('i1', {
			name: 'subpatch0',
			pos: [5, 6],
			ports: [{ uid: 'p0', type: 'OutArray', name: 'wave', inner: ['m1', 'out'] }]
		});
		seed(fc).patch({
			nodes: {
				n0: node('Oscillator', 'osc0'), // top-level node
				m1: node('Buffer', 'buffer0', 'i1'), // member of i1
				...sp.nodes
			},
			links: sp.links
		});

		// ROOT synthesized: top-level node + the scope, keyed by uid; the member excluded.
		const root = g.instances[ROOT_ID];
		expect(root, 'ROOT scope present for the canvas to render').toBeDefined();
		expect(root.members.n0).toEqual({ uid: 'n0', is_instance: false });
		expect(root.members.i1).toEqual({ uid: 'i1', is_instance: true });
		expect(root.members.m1).toBeUndefined(); // a member of i1, not top-level

		// The scope assembled from the doc.
		const i1 = g.instances.i1;
		expect(i1.name).toBe('subpatch0');
		expect(i1.pos).toEqual([5, 6]);
		expect(i1.slots).toEqual({ input: {}, output: { p0: 'ARRAY' } });
		// A port is a member like any other node — it is what the canvas draws inside the scope.
		expect(i1.members).toEqual({
			m1: { uid: 'm1', is_instance: false },
			p0: { uid: 'p0', is_instance: false }
		});

		// The synth node the canvas renders for the collapsed sub-patch reflects the wired slot.
		const synth = g.nodeById('i1');
		expect(synth?.output_slots).toEqual({ p0: 'ARRAY' });
	});

	it('a scope removed from the doc vanishes and its member returns to ROOT', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const d = seed(fc).patch({
			nodes: { m1: node('Buffer', 'buffer0', 'i1'), ...scope('i1', { name: 'sp0' }).nodes }
		});
		expect(g.instances.i1).toBeDefined();
		expect(g.instances[ROOT_ID].members.m1).toBeUndefined(); // owned by i1

		// Expand: the facade leaves the document and its member loses the `scope` naming it — the
		// exact delta the manager sends, both halves in one patch.
		d.patch({ nodes: { i1: null, m1: { scope: null } } });
		expect(g.instances.i1, 'scope dropped when removed from the doc').toBeUndefined();
		expect(g.instances[ROOT_ID].members.m1).toEqual({ uid: 'm1', is_instance: false });
	});

	it('derives a collapsed scope deep-error from a member NODE error (recursion-correct)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		// m1 exists at top level and goes into error. The bridge only ever emits `error` keyed by a
		// real NODE uid (never a scope uid) — so the scope error must be DERIVED from members.
		const d = seed(fc).node('m1', 'Buffer', 'buffer0');
		fc.emit({ event: 'error', payload: { node: 'm1', error: 'member boom' } });
		expect(g.nodeById('m1')!.error).toBe('member boom');

		// Grouping m1 into i1 (mirror writes the scope → doc reconcile) must redden the collapsed
		// sub-patch with its member's deep error, as describe_instance.error did pre-cutover.
		d.patch({ nodes: { ...scope('i1', { name: 'sp0' }).nodes, m1: { scope: 'i1' } } });
		expect(g.instances.i1.error, 'collapsed scope reflects its member deep error').toBe('member boom');

		// Clearing the member error and re-reconciling clears the scope error (no stale chip).
		fc.emit({ event: 'error', payload: { node: 'm1', error: null } });
		d.patch({ nodes: { i1: { name: 'sp0b' } } });
		expect(g.instances.i1.error, 'cleared member error clears the derived scope error').toBeNull();
	});

	it('a member runtime error live-updates the collapsed scope badge (no doc transaction)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		seed(fc).patch({
			nodes: { m1: node('Buffer', 'buffer0', 'i1'), ...scope('i1', { name: 'sp0' }).nodes }
		});
		expect(g.instances.i1.error).toBeNull();

		// A member's runtime error arrives via the `error` event (keyed by the member NODE uid). It
		// fires NO doc transaction, so the collapsed badge must be recomputed from members right here.
		fc.emit({ event: 'error', payload: { node: 'm1', error: 'runtime boom' } });
		expect(g.nodeById('m1')!.error).toBe('runtime boom');
		expect(g.instances.i1.error, 'collapsed scope reflects the member runtime error live').toBe('runtime boom');
		// …and the collapsed synth node's border reflects it (its sig includes error).
		expect(g.nodeById('i1')!.error).toBe('runtime boom');

		// Recovery clears it live too.
		fc.emit({ event: 'error', payload: { node: 'm1', error: null } });
		expect(g.instances.i1.error, 'collapsed scope clears when the member recovers').toBeNull();
	});

	it('the synth node keeps a stable reference across an unrelated doc change', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const d = seed(fc).patch({
			nodes: {
				m1: node('Buffer', 'buffer0', 'i1'),
				n0: node('Oscillator', 'osc0'),
				...scope('i1', { name: 'sp0' }).nodes
			}
		});
		const before = g.nodeById('i1');
		// A change to an UNRELATED node must not churn the sub-patch synth node identity.
		d.patch({ nodes: { n0: { name: 'osc0b' } } });
		expect(g.nodeById('i1'), 'synth node reference stable when the scope is unchanged').toBe(before);
	});
});

describe('a collapsed scope’s inline viewer, whose blob only the instance record holds', () => {
	it('answers a write, and a re-minted uid inherits nothing of it', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const sp9 = scope('i9', {
			name: 'sp9',
			ports: [{ uid: 'p9', type: 'OutArray', name: 'wave', inner: ['m9', 'out'] }]
		});
		const spec = { nodes: { m9: node('Buffer', 'buffer9', 'i9'), ...sp9.nodes }, links: sp9.links };
		const d = seed(fc).patch(spec);

		// The user gives the collapsed sub-patch's boundary slot an inline viewer and collapses it.
		g.setSlotView('i9', 'p9', { kind: 'image', collapsed: true });
		expect(slotView(g.nodeById('i9'), 'p9').kind).toBe('image');
		expect(isSlotExpanded(g.nodeById('i9'), 'p9')).toBe(false);
		// A scope uid is not a node: the engine refuses one, so the record is the whole of the state
		// and nothing is sent. (This is also why it does not survive a reload.)
		expect(fc.recordedCalls().some((c) => c.op === 'edit_node')).toBe(false);

		// An unrelated doc write re-assembles every scope from the doc, which carries no viewer blob.
		d.patch({ nodes: { m9: { name: 'buffer9b' } } });
		expect(slotView(g.nodeById('i9'), 'p9').kind, 'a survivor keeps its live view state').toBe('image');

		// Ungroup: the facade and its port leave the doc, taking the blob with them. That uid can be
		// re-minted by a later backend, and what comes back must start clean.
		d.patch({ nodes: { i9: null, p9: null, m9: { scope: null } }, links: [] });
		expect(g.instances.i9).toBeUndefined();
		d.patch(spec);
		expect(slotView(g.nodeById('i9'), 'p9').kind, 'the re-minted scope inherits no kind').toBeUndefined();
		expect(isSlotExpanded(g.nodeById('i9'), 'p9'), 'nor a collapse').toBe(true);
	});
});
