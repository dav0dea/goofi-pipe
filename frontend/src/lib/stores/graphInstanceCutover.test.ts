import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { nodesMap, instancesMap } from '$lib/crdt/graphDoc';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import * as Y from 'yjs';

/** Minimal catalog — its presence flips the store to doc-authoritative identity. */
function catalog(): NodeTypeInfo[] {
	const mk = (type: string): NodeTypeInfo => ({
		type,
		category: 'array',
		doc: '',
		available: true,
		dynamic: false,
		missing_deps: [],
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		params: {}
	});
	return [mk('Oscillator'), mk('Buffer')];
}

function seedNode(nodes: Y.Map<Y.Map<unknown>>, uid: string, type: string, name: string): void {
	const n = new Y.Map<unknown>();
	n.set('type', type);
	n.set('name', name);
	nodes.set(uid, n);
}

interface Bnd {
	bnd_id: string;
	dir: 'in' | 'out';
	dtype: string;
	name: string;
	pos?: [number, number];
	inner_node?: string;
	inner_slot?: string;
}

/** Seed an instance into the doc in the exact shape the Rust mirror (`upsert_instance`) writes it. */
function seedInstance(
	insts: Y.Map<Y.Map<unknown>>,
	uid: string,
	o: { name: string; parent?: string; def_id?: string; pos?: [number, number]; members?: Record<string, string>; interface?: Bnd[] }
): void {
	const m = new Y.Map<unknown>();
	m.set('name', o.name);
	m.set('parent', o.parent ?? ROOT_ID);
	if (o.def_id !== undefined) m.set('def_id', o.def_id);
	const p = new Y.Map<unknown>();
	p.set('x', (o.pos ?? [0, 0])[0]);
	p.set('y', (o.pos ?? [0, 0])[1]);
	m.set('pos', p);
	const mem = new Y.Map<unknown>();
	for (const [local, muid] of Object.entries(o.members ?? {})) mem.set(local, muid);
	m.set('members', mem);
	const iface = new Y.Map<Y.Map<unknown>>();
	for (const b of o.interface ?? []) {
		const bm = new Y.Map<unknown>();
		bm.set('dir', b.dir);
		bm.set('dtype', b.dtype);
		bm.set('name', b.name);
		const bp = new Y.Map<unknown>();
		bp.set('x', (b.pos ?? [0, 0])[0]);
		bp.set('y', (b.pos ?? [0, 0])[1]);
		bm.set('pos', bp);
		if (b.inner_node !== undefined) bm.set('inner_node', b.inner_node);
		if (b.inner_slot !== undefined) bm.set('inner_slot', b.inner_slot);
		iface.set(b.bnd_id, bm);
	}
	m.set('interface', iface);
	insts.set(uid, m);
}

describe('instance-forest read cutover — instances built from the doc when the catalog is present', () => {
	it('synthesizes ROOT and assembles a real instance from the doc forest', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		Y.transact(g.doc, () => {
			const nodes = nodesMap(g.doc);
			seedNode(nodes, 'n0', 'Oscillator', 'osc0'); // top-level node
			seedNode(nodes, 'm1', 'Buffer', 'buffer0'); // member of i1
			seedInstance(instancesMap(g.doc), 'i1', {
				name: 'subpatch0',
				pos: [5, 6],
				members: { buffer0: 'm1' },
				interface: [{ bnd_id: 'out0', dir: 'out', dtype: 'ARRAY', name: 'wave', inner_node: 'm1', inner_slot: 'out' }]
			});
		});

		// ROOT synthesized: top-level node + the instance, keyed by name; the member excluded.
		const root = g.instances[ROOT_ID];
		expect(root, 'ROOT scope present for the canvas to render').toBeDefined();
		expect(root.members.osc0).toEqual({ uid: 'n0', is_instance: false });
		expect(root.members.subpatch0).toEqual({ uid: 'i1', is_instance: true });
		expect(root.members.buffer0).toBeUndefined(); // a member of i1, not top-level

		// The real instance assembled from the doc.
		const i1 = g.instances.i1;
		expect(i1.name).toBe('subpatch0');
		expect(i1.kind).toBe('unique');
		expect(i1.pos).toEqual([5, 6]);
		expect(i1.slots).toEqual({ input: {}, output: { out0: 'ARRAY' } });
		expect(i1.members).toEqual({ buffer0: { uid: 'm1', is_instance: false } });

		// The synth node the canvas renders for the collapsed sub-patch reflects the wired slot.
		const synth = g.nodeById('i1');
		expect(synth?.output_slots).toEqual({ out0: 'ARRAY' });
	});

	it('an instance removed from the doc vanishes and its member returns to ROOT', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		Y.transact(g.doc, () => {
			seedNode(nodesMap(g.doc), 'm1', 'Buffer', 'buffer0');
			seedInstance(instancesMap(g.doc), 'i1', { name: 'sp0', members: { buffer0: 'm1' } });
		});
		expect(g.instances.i1).toBeDefined();
		expect(g.instances[ROOT_ID].members.buffer0).toBeUndefined(); // owned by i1

		// Expand: the instance is removed from the doc (its member m1 becomes top-level).
		Y.transact(g.doc, () => instancesMap(g.doc).delete('i1'));
		expect(g.instances.i1, 'instance dropped when removed from the doc').toBeUndefined();
		expect(g.instances[ROOT_ID].members.buffer0).toEqual({ uid: 'm1', is_instance: false });
	});

	it('ignores a subpatch_changed event when the catalog is authoritative', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		Y.transact(g.doc, () => seedNode(nodesMap(g.doc), 'n0', 'Oscillator', 'osc0'));

		// A phantom instance in a subpatch_changed snapshot must NOT appear — the doc owns the forest.
		fc.emit({
			event: 'subpatch_changed',
			// Minimal snapshot — its only job is to prove the handler early-breaks when the catalog is
			// present, so the irrelevant required GraphSnapshot fields are cast away.
			payload: {
				nodes: [],
				links: [],
				instances: {
					ghost: {
						uid: 'ghost',
						name: 'ghost0',
						kind: 'unique',
						def_id: null,
						parent: ROOT_ID,
						interface: {},
						pos: [0, 0],
						members: {},
						slots: { input: {}, output: {} },
						siblings: [],
						error: null,
						viewers: {}
					}
				}
			} as unknown as GraphSnapshot
		});
		expect(g.instances.ghost).toBeUndefined();
	});

	it('derives a collapsed instance deep-error from a member NODE error (recursion-correct)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		// m1 exists at top level and goes into error. The bridge only ever emits `error` keyed by a
		// real NODE uid (never an instance uid) — so the instance error must be DERIVED from members,
		// not overlaid from a (never-sent) instance-keyed event.
		Y.transact(g.doc, () => seedNode(nodesMap(g.doc), 'm1', 'Buffer', 'buffer0'));
		fc.emit({ event: 'error', payload: { node: 'm1', error: 'member boom' } });
		expect(g.nodeById('m1')!.error).toBe('member boom');

		// Grouping m1 into i1 (mirror writes the instance → doc reconcile) must redden the collapsed
		// sub-patch with its member's deep error, as describe_instance.error did pre-cutover.
		Y.transact(g.doc, () => seedInstance(instancesMap(g.doc), 'i1', { name: 'sp0', members: { buffer0: 'm1' } }));
		expect(g.instances.i1.error, 'collapsed instance reflects its member deep error').toBe('member boom');

		// Clearing the member error and re-reconciling clears the instance error (no stale chip).
		fc.emit({ event: 'error', payload: { node: 'm1', error: null } });
		Y.transact(g.doc, () => instancesMap(g.doc).get('i1')!.set('name', 'sp0b'));
		expect(g.instances.i1.error, 'cleared member error clears the derived instance error').toBeNull();
	});

	it('a member runtime error live-updates the collapsed instance badge (no doc transaction)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		Y.transact(g.doc, () => {
			seedNode(nodesMap(g.doc), 'm1', 'Buffer', 'buffer0');
			seedInstance(instancesMap(g.doc), 'i1', { name: 'sp0', members: { buffer0: 'm1' } });
		});
		expect(g.instances.i1.error).toBeNull();

		// A member's runtime error arrives via the `error` event (keyed by the member NODE uid — the
		// only error events the bridge sends). It fires NO doc transaction, so the collapsed instance
		// badge must be recomputed from members right here, not only on the next structural edit.
		fc.emit({ event: 'error', payload: { node: 'm1', error: 'runtime boom' } });
		expect(g.nodeById('m1')!.error).toBe('runtime boom');
		expect(g.instances.i1.error, 'collapsed instance reflects the member runtime error live').toBe('runtime boom');
		// …and the collapsed synth node's border reflects it (its sig includes error).
		expect(g.nodeById('i1')!.error).toBe('runtime boom');

		// Recovery clears it live too.
		fc.emit({ event: 'error', payload: { node: 'm1', error: null } });
		expect(g.instances.i1.error, 'collapsed instance clears when the member recovers').toBeNull();
	});

	it('the synth node keeps a stable reference across an unrelated doc change', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		Y.transact(g.doc, () => {
			seedNode(nodesMap(g.doc), 'm1', 'Buffer', 'buffer0');
			seedNode(nodesMap(g.doc), 'n0', 'Oscillator', 'osc0');
			seedInstance(instancesMap(g.doc), 'i1', { name: 'sp0', members: { buffer0: 'm1' } });
		});
		const before = g.nodeById('i1');
		// A change to an UNRELATED node must not churn the sub-patch synth node identity.
		Y.transact(g.doc, () => nodesMap(g.doc).get('n0')!.set('name', 'osc0b'));
		expect(g.nodeById('i1'), 'synth node reference stable when the instance is unchanged').toBe(before);
	});
});
