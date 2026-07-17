import { describe, it, expect } from 'vitest';
import { assembleInstances, assembleInstance, assembleRoot } from './instanceAssembly';
import type { InstanceView } from './graphDoc';
import { ROOT_ID } from '$lib/editor/subpatchScene';

/** A doc instance view in the exact shape `instanceViews(doc)` returns. */
function inst(over: Partial<InstanceView> & { uid: string }): InstanceView {
	return {
		name: over.name ?? over.uid,
		def_id: over.def_id,
		parent: over.parent ?? ROOT_ID,
		pos: over.pos ?? [0, 0],
		members: over.members ?? {},
		interface: over.interface ?? [],
		uid: over.uid
	};
}

describe('instanceAssembly — real instance from a doc view', () => {
	it('derives kind/def_id/parent/slots/interface/members/viewers for a unique instance', () => {
		const view = inst({
			uid: 'i1',
			name: 'subpatch0',
			pos: [5, 6],
			members: { buffer0: 'm1' },
			interface: [
				{
					bnd_id: 'out0',
					dir: 'out',
					dtype: 'ARRAY',
					name: 'wave',
					pos: [1, 2],
					inner_node: 'm1',
					inner_slot: 'out'
				}
			]
		});
		const info = assembleInstance(view, new Set(['i1']), [view], null);

		expect(info.uid).toBe('i1');
		expect(info.name).toBe('subpatch0');
		expect(info.kind).toBe('unique'); // no def_id in the doc → unique
		expect(info.def_id).toBeNull();
		expect(info.parent).toBe(ROOT_ID); // top-level real instance parents to ROOT_ID, not null
		expect(info.pos).toEqual([5, 6]);
		// Interface maps the boundary array → Record<bnd_id, SubPatchPort>.
		expect(info.interface.out0).toEqual({
			dir: 'out',
			dtype: 'ARRAY',
			inner_node: 'm1',
			inner_slot: 'out',
			pos: [1, 2],
			name: 'wave'
		});
		// A WIRED output boundary (inner_node != null) becomes an output slot.
		expect(info.slots).toEqual({ input: {}, output: { out0: 'ARRAY' } });
		// members local→uid gains is_instance (m1 is a plain node here).
		expect(info.members).toEqual({ buffer0: { uid: 'm1', is_instance: false } });
		expect(info.siblings).toEqual([]);
		expect(info.viewers).toEqual({}); // instance viewers are a backend stub ({} end-to-end)
		expect(info.error).toBeNull();
	});

	it('an UNWIRED boundary renders in the interface but is not a slot', () => {
		const view = inst({
			uid: 'i1',
			interface: [
				{ bnd_id: 'in0', dir: 'in', dtype: 'ARRAY', name: 'x', pos: [0, 0] } // no inner_node
			]
		});
		const info = assembleInstance(view, new Set(['i1']), [view], null);
		expect(info.interface.in0.inner_node).toBeNull();
		expect(info.slots).toEqual({ input: {}, output: {} }); // unwired → no slot
	});

	it('splits wired boundaries into input vs output slots by dir', () => {
		const view = inst({
			uid: 'i1',
			interface: [
				{ bnd_id: 'in0', dir: 'in', dtype: 'ARRAY', name: 'x', pos: [0, 0], inner_node: 'm1', inner_slot: 'data' },
				{ bnd_id: 'out0', dir: 'out', dtype: 'STRING', name: 'y', pos: [0, 0], inner_node: 'm1', inner_slot: 'out' }
			]
		});
		const info = assembleInstance(view, new Set(['i1']), [view], null);
		expect(info.slots).toEqual({ input: { in0: 'ARRAY' }, output: { out0: 'STRING' } });
	});

	it('a member that is itself an instance gets is_instance=true', () => {
		const parent = inst({ uid: 'i1', members: { nested0: 'i2' } });
		const nested = inst({ uid: 'i2', parent: 'i1' });
		const info = assembleInstance(parent, new Set(['i1', 'i2']), [parent, nested], null);
		expect(info.members).toEqual({ nested0: { uid: 'i2', is_instance: true } });
	});

	it('overlays the runtime error from the lookup', () => {
		const view = inst({ uid: 'i1' });
		const info = assembleInstance(view, new Set(['i1']), [view], 'member boom');
		expect(info.error).toBe('member boom');
	});
});

describe('instanceAssembly — shared family siblings', () => {
	it('two instances sharing a def_id reference each other; kind=shared', () => {
		const a = inst({ uid: 'ia', def_id: 'defX' });
		const b = inst({ uid: 'ib', def_id: 'defX' });
		const all = [a, b];
		const ia = assembleInstance(a, new Set(['ia', 'ib']), all, null);
		const ib = assembleInstance(b, new Set(['ia', 'ib']), all, null);
		expect(ia.kind).toBe('shared');
		expect(ia.def_id).toBe('defX');
		expect(ia.siblings).toEqual(['ib']);
		expect(ib.siblings).toEqual(['ia']);
	});

	it('a unique instance (no def_id) has no siblings even if another unique exists', () => {
		const a = inst({ uid: 'ia' });
		const b = inst({ uid: 'ib' });
		const info = assembleInstance(a, new Set(['ia', 'ib']), [a, b], null);
		expect(info.siblings).toEqual([]);
	});
});

describe('instanceAssembly — synthetic ROOT', () => {
	it('ROOT members are the top-level nodes + instances, keyed by name; members of an instance are excluded', () => {
		const i1 = inst({ uid: 'i1', name: 'subpatch0', members: { buffer0: 'm1' } });
		const nodes = [
			{ uid: 'n0', name: 'osc0' }, // top-level node
			{ uid: 'm1', name: 'buffer0' } // MEMBER of i1 — must NOT be in ROOT
		];
		const root = assembleRoot(nodes, [i1]);

		expect(root.uid).toBe(ROOT_ID);
		expect(root.name).toBe('root');
		expect(root.kind).toBe('unique');
		expect(root.def_id).toBeNull();
		expect(root.parent).toBeNull(); // ROOT itself parents to null (unlike real top-level instances)
		expect(root.error).toBeNull();
		// osc0 (top-level node) + subpatch0 (top-level instance); buffer0 (a member) excluded.
		expect(root.members).toEqual({
			osc0: { uid: 'n0', is_instance: false },
			subpatch0: { uid: 'i1', is_instance: true }
		});
	});

	it('a nested instance is excluded from ROOT (it is a member of its parent)', () => {
		const i1 = inst({ uid: 'i1', name: 'outer', members: { inner0: 'i2' } });
		const i2 = inst({ uid: 'i2', name: 'inner', parent: 'i1' });
		const root = assembleRoot([], [i1, i2]);
		// Only the outer instance is top-level; inner is i1's member.
		expect(Object.keys(root.members)).toEqual(['outer']);
		expect(root.members.outer).toEqual({ uid: 'i1', is_instance: true });
	});
});

describe('instanceAssembly — assembleInstances (full map incl. ROOT)', () => {
	it('returns ROOT + every real instance, wiring the runtime-error lookup', () => {
		const i1 = inst({ uid: 'i1', name: 'sp0', members: { buffer0: 'm1' } });
		const nodes = [
			{ uid: 'n0', name: 'osc0' },
			{ uid: 'm1', name: 'buffer0' }
		];
		const map = assembleInstances([i1], nodes, (uid) => (uid === 'i1' ? 'deep error' : null));

		expect(Object.keys(map).sort()).toEqual([ROOT_ID, 'i1'].sort());
		expect(map[ROOT_ID].members.osc0).toEqual({ uid: 'n0', is_instance: false });
		expect(map[ROOT_ID].members.sp0).toEqual({ uid: 'i1', is_instance: true });
		expect(map.i1.error).toBe('deep error');
		expect(map.i1.members.buffer0).toEqual({ uid: 'm1', is_instance: false });
	});
});
