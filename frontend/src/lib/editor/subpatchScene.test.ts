import { describe, it, expect } from 'vitest';
import {
	buildMemberIndex,
	parentOf,
	childrenOfScope,
	drawEndpoint,
	boundaryNodeId,
	isBoundaryNodeId,
	parseBoundaryNodeId,
	type SceneInstance
} from './subpatchScene';

// outer > [nodeA, inner]; inner > [leaf]. inner exposes leaf.out as boundary 'iout';
// outer exposes inner.iout as boundary 'oout' (a 2-level chain).
function fixture(): Record<string, SceneInstance> {
	return {
		outer: {
			parent: null,
			members: {
				nodeA: { uid: 'a', is_instance: false },
				inner: { uid: 'inner', is_instance: true }
			},
			interface: {
				oout: { dir: 'out', inner_node: 'inner', inner_slot: 'iout' }
			}
		},
		inner: {
			parent: 'outer',
			members: { leaf: { uid: 'leaf', is_instance: false } },
			interface: {
				iout: { dir: 'out', inner_node: 'leaf', inner_slot: 'out' }
			}
		}
	};
}

describe('parentOf', () => {
	it('walks node -> instance -> instance -> null', () => {
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		expect(parentOf('a', idx)).toBe('outer'); // node directly in outer
		expect(parentOf('leaf', idx)).toBe('inner'); // node in nested inner
		expect(parentOf('inner', idx)).toBe('outer'); // nested instance's parent
		expect(parentOf('outer', idx)).toBe(null); // root
		expect(parentOf('top', idx)).toBe(null); // a top-level node (not a member)
	});
});

describe('childrenOfScope', () => {
	it('returns only the DIRECT children of a scope (nested instance not leaked to root)', () => {
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		const nodeUids = ['a', 'leaf', 'top'];

		const root = childrenOfScope(null, insts, nodeUids, idx);
		expect(root.nodeUids).toEqual(['top']); // 'a' is inside outer; 'leaf' inside inner
		expect(root.instUids).toEqual(['outer']); // 'inner' is NOT a root child

		const inOuter = childrenOfScope('outer', insts, nodeUids, idx);
		expect(inOuter.nodeUids).toEqual(['a']);
		expect(inOuter.instUids).toEqual(['inner']); // nested instance renders inside outer

		const inInner = childrenOfScope('inner', insts, nodeUids, idx);
		expect(inInner.nodeUids).toEqual(['leaf']);
		expect(inInner.instUids).toEqual([]);
	});
});

describe('drawEndpoint', () => {
	it('single-level: a top-level node at root is the identity', () => {
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		expect(drawEndpoint('top', 'out', 'out', null, insts, idx)).toEqual({ node: 'top', handle: 'out' });
	});

	it('single-level: a member of a top-level instance climbs ONE hop to its boundary', () => {
		// At root scope, leaf is two levels down but inner is a child of outer; from the
		// perspective of entering `inner`, leaf.out -> the inner boundary 'iout'.
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		expect(drawEndpoint('leaf', 'out', 'out', 'inner', insts, idx)).toEqual({ node: 'leaf', handle: 'out' });
		// leaf is a direct child of inner, so inside inner it draws as itself.
	});

	it('multi-level: walks up to the nearest visible ancestor boundary at the root scope', () => {
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		// leaf.out at the ROOT scope: leaf -> inner.iout -> outer.oout (the collapsed outer node's port)
		expect(drawEndpoint('leaf', 'out', 'out', null, insts, idx)).toEqual({ node: 'outer', handle: 'oout' });
		// a direct child of outer at the outer scope draws as itself
		expect(drawEndpoint('a', 'x', 'out', 'outer', insts, idx)).toEqual({ node: 'a', handle: 'x' });
	});

	it('returns null when the slot is not exposed up the chain (purely internal link)', () => {
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		// leaf has no boundary for slot 'private' -> not exposed -> hidden at the root scope
		expect(drawEndpoint('leaf', 'private', 'out', null, insts, idx)).toBe(null);
	});

	it('returns null when the endpoint is outside the entered scope subtree', () => {
		const insts = fixture();
		const idx = buildMemberIndex(insts);
		// 'top' (a root node) viewed from inside 'inner' is outside that subtree
		expect(drawEndpoint('top', 'out', 'out', 'inner', insts, idx)).toBe(null);
	});
});

describe('boundary node id codec', () => {
	it('round-trips an (instId, boundary) pair within the entered scope', () => {
		const id = boundaryNodeId('outer', 'oout');
		expect(isBoundaryNodeId(id)).toBe(true);
		expect(parseBoundaryNodeId(id, 'outer')).toBe('oout');
	});

	it('isBoundaryNodeId is scope-agnostic (any boundary pill id)', () => {
		expect(isBoundaryNodeId(boundaryNodeId('inner', 'iout'))).toBe(true);
		expect(isBoundaryNodeId('plain-node-uid')).toBe(false);
	});

	it('parse returns null for a non-boundary id', () => {
		expect(parseBoundaryNodeId('plain-node-uid', 'outer')).toBe(null);
	});

	it('parse returns null for a boundary id from ANOTHER instance (scope-checked)', () => {
		// A pill belonging to `inner` must not be mis-parsed as one of `outer`'s — the
		// old length-slice returned a garbage suffix instead of null.
		const innerPill = boundaryNodeId('inner', 'iout');
		expect(parseBoundaryNodeId(innerPill, 'outer')).toBe(null);
	});

	it('parse returns null when no scope is entered', () => {
		expect(parseBoundaryNodeId(boundaryNodeId('outer', 'oout'), null)).toBe(null);
	});
});
