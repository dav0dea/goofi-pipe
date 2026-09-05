import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { SCOPE_TYPE, BOUNDARY_SLOT, boundaryType } from '$lib/api/vocab';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { NodeTypeInfo, GraphSnapshot } from '$lib/api/control';
import { slotView, isSlotExpanded } from '$lib/viewers/inlineView';
import { liveCatalogue } from '$lib/inspector/expr/catalogue';
import { typeInfo } from '$lib/test/typeInfo';

/** Minimal catalog — its presence flips the store to doc-authoritative identity. */
function catalog(): NodeTypeInfo[] {
	const mk = (type: string): NodeTypeInfo => typeInfo({
		type,
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		params: {}
	});
	return [mk('signal:Oscillator'), mk('signal:Buffer')];
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
	it('assembles a scope from the doc forest, as one node record among the rest', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const sp = scope('i1', {
			name: 'subpatch0',
			pos: [5, 6],
			ports: [
				{ uid: 'p0', type: 'OutArray', name: 'wave', inner: ['m1', 'out'] },
				{ uid: 'p1', type: 'InArray', name: 'feed' } // authored, not yet wired inside
			]
		});
		seed(fc).patch({
			nodes: {
				n0: node('signal:Oscillator', 'osc0'), // top-level node
				m1: node('signal:Buffer', 'buffer0', 'i1'), // member of i1
				...sp.nodes
			},
			links: sp.links
		});

		// Membership rides the record, so ROOT is simply the scope a top-level record names.
		const members = (scope: string) => g.nodes.filter((n) => n.scope === scope).map((n) => n.uid);
		expect.soft(members(ROOT_ID)).toEqual(['n0', 'i1']);
		// A port is a member like any other node — it is what the canvas draws inside the scope.
		expect.soft(members('i1')).toEqual(['m1', 'p0', 'p1']);

		// The facade is one node in the ONE list, with the position and name the doc gave it.
		const i1 = g.nodeById('i1')!;
		expect(i1.name).toBe('subpatch0');
		expect(i1.pos).toEqual([5, 6]);
		expect(i1.subpatch).toEqual({ memberCount: 3 });

		// It exposes ONE slot per port, by direction. The inner wire is a separate question:
		// authoring `p1` is what gives the parent's facade its input.
		expect.soft(i1.output_slots).toEqual({ p0: 'ARRAY' });
		expect.soft(i1.input_slots).toEqual({ p1: 'ARRAY' });
		expect.soft(i1.slot_labels).toEqual({ p0: 'wave', p1: 'feed' });
	});

	it('a scope removed from the doc vanishes and its member returns to ROOT', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const d = seed(fc).patch({
			nodes: { m1: node('signal:Buffer', 'buffer0', 'i1'), ...scope('i1', { name: 'sp0' }).nodes }
		});
		expect(g.nodeById('i1')?.subpatch).toBeDefined();
		expect(g.nodeById('m1')!.scope).toBe('i1');

		// Expand: the facade leaves the document and its member loses the `scope` naming it — the
		// exact delta the manager sends, both halves in one patch.
		d.patch({ nodes: { i1: null, m1: { scope: null } } });
		expect(g.nodeById('i1'), 'scope dropped when removed from the doc').toBeNull();
		expect(g.nodeById('m1')!.scope).toBe(ROOT_ID);
	});

	it('a facade keeps a stable reference across an unrelated doc change', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const d = seed(fc).patch({
			nodes: {
				m1: node('signal:Buffer', 'buffer0', 'i1'),
				n0: node('signal:Oscillator', 'osc0'),
				...scope('i1', { name: 'sp0' }).nodes
			}
		});
		const before = g.nodeById('i1');
		// A change to an UNRELATED node must not churn the sub-patch synth node identity.
		d.patch({ nodes: { n0: { name: 'osc0b' } } });
		expect(g.nodeById('i1'), 'facade reference stable when the scope is unchanged').toBe(before);
	});
});

describe('a collapsed scope’s inline viewer, which is a node’s inline viewer', () => {
	it('goes out as edit_node, rides the doc back, and a re-minted uid inherits nothing of it', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const sp9 = scope('i9', {
			name: 'sp9',
			ports: [{ uid: 'p9', type: 'OutArray', name: 'wave', inner: ['m9', 'out'] }]
		});
		const spec = { nodes: { m9: node('signal:Buffer', 'buffer9', 'i9'), ...sp9.nodes }, links: sp9.links };
		// The snapshot's runtime overlay lands BEFORE the doc materializes the nodes it names — the
		// two ride separate channels in no defined order.
		fc.emit({
			event: 'hello',
			payload: {
				runtime: { i9: { stage: 'ready', error: 'a member of it was failing' } },
				save_path: null,
				unsaved_changes: false,
				instance_id: 'sess1',
				viewpoint: null
			} as unknown as GraphSnapshot
		});
		const d = seed(fc).patch(spec);
		expect(g.nodeById('i9')!.error, 'the overlay reaches a node that arrives after it').toBe(
			'a member of it was failing'
		);

		// The user gives the collapsed sub-patch's boundary slot an inline viewer and collapses it.
		g.setSlotView('i9', 'p9', { kind: 'image', collapsed: true });
		// A facade is a node here and in the engine, so the write is the op a leaf's write is —
		// which is what makes it undoable, replicated to a second tab, and saved with the patch.
		const sent = fc.recordedCalls().find((c) => c.op === 'node edit');
		expect(sent?.payload).toMatchObject({ node: 'i9', viewer: [{ slot: 'p9', kind: 'image' }] });

		// …and the state it shows comes BACK through the doc, exactly as a leaf's does.
		expect(slotView(g.nodeById('i9'), 'p9').kind, 'nothing until the manager answers').toBeUndefined();
		d.patch({ nodes: { i9: { viewers: JSON.stringify({ p9: { kind: 'image', collapsed: true } }) } } });
		expect(slotView(g.nodeById('i9'), 'p9').kind).toBe('image');
		expect(isSlotExpanded(g.nodeById('i9'), 'p9')).toBe(false);

		// An unrelated doc write re-assembles every scope; the blob is in the doc, so it survives.
		d.patch({ nodes: { m9: { name: 'buffer9b' } } });
		expect(slotView(g.nodeById('i9'), 'p9').kind, 'a survivor keeps its view state').toBe('image');

		// Ungroup: the facade and its port leave the doc, taking the blob with them. That uid can be
		// re-minted by a later backend, and what comes back must start clean.
		d.patch({ nodes: { i9: null, p9: null, m9: { scope: null } }, links: [] });
		expect(g.nodeById('i9')).toBeNull();
		d.patch(spec);
		expect(slotView(g.nodeById('i9'), 'p9').kind, 'the re-minted scope inherits no kind').toBeUndefined();
		expect(isSlotExpanded(g.nodeById('i9'), 'p9'), 'nor a collapse').toBe(true);
		expect(g.nodeById('i9')!.error, 'nor the badge that overlay handed the first one').toBeNull();
	});
});

describe('the expression catalogue — what nd() can name', () => {
	it('offers a sub-patch by name and its ports by name, beside the leaves', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const sp = scope('i2', {
			name: 'subpatch0',
			ports: [
				{ uid: 'p2', type: 'OutArray', name: 'wave', inner: ['m2', 'out'] },
				{ uid: 'p3', type: 'InArray', name: 'feed' }
			]
		});
		seed(fc).patch({
			nodes: { n1: node('signal:Oscillator', 'osc0'), m2: node('signal:Buffer', 'buffer0', 'i2'), ...sp.nodes },
			links: sp.links
		});

		const names = liveCatalogue(g).nodes.map((n) => n.name);
		// A facade and a port are things `nd()` resolves, so a completion that cannot offer them
		// hides half the patch from the one surface that exists to say what is nameable.
		expect(names).toContain('subpatch0');
		expect(names).toContain('wave');
		expect(names).toContain('osc0');

		// A facade's slots are keyed by port uid and ADDRESSED by the port's name, so the name is
		// what is offered — a uid would complete to an expression the resolver refuses.
		const facade = liveCatalogue(g).nodes.find((n) => n.name === 'subpatch0');
		expect(facade?.slots.map((s) => s.name)).toEqual(['wave']);
	});
});
