import { describe, it, expect } from 'vitest';
import { assembleInstances, assembleInstance, assembleRoot } from './instanceAssembly';
import type { InstanceView } from './graphDoc';
import { ROOT_ID } from '$lib/editor/subpatchScene';

/** A doc scope view in the exact shape `instanceViews(doc)` returns (flat model: members keyed by
 * uid → is_instance; no def_id/kind/siblings). */
function inst(over: Partial<InstanceView> & { uid: string }): InstanceView {
	return {
		name: over.name ?? over.uid,
		parent: over.parent ?? ROOT_ID,
		pos: over.pos ?? [0, 0],
		members: over.members ?? {},
		interface: over.interface ?? [],
		uid: over.uid
	};
}

describe('instanceAssembly — a scope from a doc view', () => {
	it('derives parent/slots/interface/members/viewers for a scope', () => {
		const view = inst({
			uid: 'i1',
			name: 'subpatch0',
			pos: [5, 6],
			members: { m1: false },
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
		const info = assembleInstance(view, new Set(['i1']), null);

		expect(info.uid).toBe('i1');
		expect(info.name).toBe('subpatch0');
		expect(info.parent).toBe(ROOT_ID); // top-level scope parents to ROOT_ID, not null
		expect(info.pos).toEqual([5, 6]);
		// Interface maps the stub array → Record<stub_id, SubPatchPort>.
		expect(info.interface.out0).toEqual({
			dir: 'out',
			dtype: 'ARRAY',
			inner_node: 'm1',
			inner_slot: 'out',
			pos: [1, 2],
			name: 'wave'
		});
		// members keyed by uid → {uid, is_instance} (m1 is a plain node here).
		expect(info.members).toEqual({ m1: { uid: 'm1', is_instance: false } });
		expect(info.viewers).toEqual({});
		expect(info.error).toBeNull();
	});

	it('an UNWIRED stub is a port like any other — the inner wire is a separate question', () => {
		const view = inst({
			uid: 'i1',
			interface: [
				{ bnd_id: 'in0', dir: 'in', dtype: 'ARRAY', name: 'x', pos: [0, 0] } // no inner_node
			]
		});
		const info = assembleInstance(view, new Set(['i1']), null);
		expect(info.interface.in0.inner_node).toBeNull();
		expect(info.interface.in0.dir).toBe('in');
	});

	it('a member that is itself a scope gets is_instance=true', () => {
		const parent = inst({ uid: 'i1', members: { i2: true } });
		const info = assembleInstance(parent, new Set(['i1', 'i2']), null);
		expect(info.members).toEqual({ i2: { uid: 'i2', is_instance: true } });
	});

	it('overlays the runtime error from the lookup', () => {
		const view = inst({ uid: 'i1' });
		const info = assembleInstance(view, new Set(['i1']), 'member boom');
		expect(info.error).toBe('member boom');
	});
});

describe('instanceAssembly — synthetic ROOT', () => {
	it('ROOT members are the top-level nodes + scopes, keyed by uid; members of a scope are excluded', () => {
		const i1 = inst({ uid: 'i1', name: 'subpatch0', members: { m1: false } });
		const nodes = [
			{ uid: 'n0', name: 'osc0' }, // top-level node
			{ uid: 'm1', name: 'buffer0' } // MEMBER of i1 — must NOT be in ROOT
		];
		const root = assembleRoot(nodes, [i1]);

		expect(root.uid).toBe(ROOT_ID);
		expect(root.name).toBe('root');
		expect(root.parent).toBeNull(); // ROOT itself parents to null (unlike real top-level scopes)
		expect(root.error).toBeNull();
		// n0 (top-level node) + i1 (top-level scope); m1 (a member) excluded — keyed by uid.
		expect(root.members).toEqual({
			n0: { uid: 'n0', is_instance: false },
			i1: { uid: 'i1', is_instance: true }
		});
	});

	it('a nested scope is excluded from ROOT (it is a member of its parent)', () => {
		const i1 = inst({ uid: 'i1', name: 'outer', members: { i2: true } });
		const i2 = inst({ uid: 'i2', name: 'inner', parent: 'i1' });
		const root = assembleRoot([], [i1, i2]);
		// Only the outer scope is top-level; inner is i1's member.
		expect(Object.keys(root.members)).toEqual(['i1']);
		expect(root.members.i1).toEqual({ uid: 'i1', is_instance: true });
	});
});

describe('instanceAssembly — assembleInstances (full map incl. ROOT)', () => {
	it('returns ROOT + every scope, deriving each scope error from its member nodes', () => {
		const i1 = inst({ uid: 'i1', name: 'sp0', members: { m1: false } });
		const nodes = [
			{ uid: 'n0', name: 'osc0' },
			{ uid: 'm1', name: 'buffer0' }
		];
		// The lookup answers per NODE uid; the scope's deep error is derived from its member m1.
		const map = assembleInstances([i1], nodes, (uid) => (uid === 'm1' ? 'member boom' : null));

		expect(Object.keys(map).sort()).toEqual([ROOT_ID, 'i1'].sort());
		expect(map[ROOT_ID].members.n0).toEqual({ uid: 'n0', is_instance: false });
		expect(map[ROOT_ID].members.i1).toEqual({ uid: 'i1', is_instance: true });
		expect(map.i1.error).toBe('member boom');
		expect(map.i1.members.m1).toEqual({ uid: 'm1', is_instance: false });
	});

	it('derives a nested scope error up through the parent (recursion-correct)', () => {
		// i1 contains nested scope i2; i2 contains member node m2 which is in error. i1's deep
		// error must surface m2's error through the two levels.
		const i2 = inst({ uid: 'i2', name: 'inner', parent: 'i1', members: { m2: false } });
		const i1 = inst({ uid: 'i1', name: 'outer', members: { i2: true } });
		const map = assembleInstances([i1, i2], [{ uid: 'm2', name: 'buf' }], (uid) =>
			uid === 'm2' ? 'deep boom' : null
		);
		expect(map.i2.error).toBe('deep boom');
		expect(map.i1.error, 'error propagates up through the nested scope').toBe('deep boom');
	});
});
