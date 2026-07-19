import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { graphExecutors } from './graphExecutors';
import { history, type Action, type ExecutorDeps, type NavContext } from './history.svelte';
import { collectPanels } from '$lib/workspace/model';
import { setInlineKind, rawInlineView } from '$lib/viewers/inlineView.svelte';
import { ui } from '$lib/stores/ui.svelte';
import type { NodeInstanceInfo, LinkInfo, NodeTypeInfo } from '$lib/api/control';
import * as Y from 'yjs';
import { linksArray, paramValue, nodeView, docParams, setParamExpr, nodesMap, instancesMap } from '$lib/crdt/graphDoc';
import { ROOT_ID } from '$lib/editor/subpatchScene';

/** Minimal catalog — its presence flips the store to doc-authoritative node/instance identity. */
function catalog(): NodeTypeInfo[] {
	const mk = (type: string): NodeTypeInfo => ({
		type,
		category: 'inputs',
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

/** Seed a node into the store's CRDT doc so a param leaf-write (or a doc-authoritative
 * node read) targeting it lands. Defaults preserve the type='Oscillator', name=uid shape. */
function docAddNode(g: GraphStore, uid: string, type = 'Oscillator', name = uid): void {
	const nodes = g.doc.getMap('nodes') as Y.Map<Y.Map<unknown>>;
	if (nodes.get(uid)) return;
	const n = new Y.Map<unknown>();
	n.set('type', type);
	n.set('name', name);
	nodes.set(uid, n);
}

/** Remove a node from the store's CRDT doc (mirror of the manager dropping it on delete). */
function docRemoveNode(g: GraphStore, uid: string): void {
	(g.doc.getMap('nodes') as Y.Map<Y.Map<unknown>>).delete(uid);
}

/** Seed a node into an arbitrary `nodes` map (doc-authoritative instance seeding). */
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

/** Seed a scope into the doc in the flat shape the Rust mirror writes: members keyed by uid →
 * {is_instance}, stubs keyed by id. */
function seedInstance(
	insts: Y.Map<Y.Map<unknown>>,
	uid: string,
	o: { name: string; parent?: string; pos?: [number, number]; members?: Record<string, boolean>; stubs?: Bnd[] }
): void {
	const m = new Y.Map<unknown>();
	m.set('name', o.name);
	m.set('parent', o.parent ?? ROOT_ID);
	const p = new Y.Map<unknown>();
	p.set('x', (o.pos ?? [0, 0])[0]);
	p.set('y', (o.pos ?? [0, 0])[1]);
	m.set('pos', p);
	const mem = new Y.Map<Y.Map<unknown>>();
	for (const [muid, isInst] of Object.entries(o.members ?? {})) {
		const e = new Y.Map<unknown>();
		e.set('is_instance', isInst);
		mem.set(muid, e);
	}
	m.set('members', mem);
	const stubs = new Y.Map<Y.Map<unknown>>();
	for (const b of o.stubs ?? []) {
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
		stubs.set(b.bnd_id, bm);
	}
	m.set('stubs', stubs);
	insts.set(uid, m);
}

const EMPTY_CTX: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

/** Seed a link into the store's CRDT doc — links are read from the doc (Phase 2), so a test
 * simulates the manager's link mirror by writing the doc (Yjs observers fire synchronously). */
function docAddLink(g: GraphStore, link: LinkInfo): void {
	const m = new Y.Map<unknown>();
	m.set('node_out', link.node_out);
	m.set('slot_out', link.slot_out);
	m.set('node_in', link.node_in);
	m.set('slot_in', link.slot_in);
	linksArray(g.doc).push([m]);
}
/** Clear all links from the store's CRDT doc (mirror of the manager tearing links down). */
function docClearLinks(g: GraphStore): void {
	const a = linksArray(g.doc);
	a.delete(0, a.length);
}
/** Seed/update an instance's position in the doc — instance placement is doc-owned (Phase 2). */
function docSetInstancePos(g: GraphStore, uid: string, pos: [number, number]): void {
	const instances = g.doc.getMap('instances') as Y.Map<Y.Map<unknown>>;
	let inst = instances.get(uid);
	if (!inst) {
		inst = new Y.Map<unknown>();
		instances.set(uid, inst);
	}
	let p = inst.get('pos') as Y.Map<unknown> | undefined;
	if (!p) {
		p = new Y.Map<unknown>();
		inst.set('pos', p);
	}
	p.set('x', pos[0]);
	p.set('y', pos[1]);
}

function nodeInfo(name: string, type = 'Oscillator'): NodeInstanceInfo {
	return {
		uid: name,
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

function deps(fc: FakeControl, g: GraphStore): ExecutorDeps {
	return { control: fc, graph: g, workspace: workspace() };
}

describe('FakeControl', () => {
	it('records calls and fans out emitted events; never auto-emits', async () => {
		const fc = new FakeControl();
		let got: unknown = null;
		fc.on((ev) => (got = ev));
		await fc.call('add_node', { type: 'X' });
		expect(fc.recordedCalls()).toEqual([{ op: 'add_node', payload: { type: 'X' } }]);
		expect(got).toBe(null); // no auto-emit
		fc.emit({ event: 'node_added', payload: nodeInfo('x0') });
		expect((got as { event: string }).event).toBe('node_added');
	});

	it('setCallResult makes a call resolve to a chosen value', async () => {
		const fc = new FakeControl();
		fc.setCallResult('add_node', 'osc0');
		expect(await fc.call('add_node', {})).toBe('osc0');
	});
});

describe('graph executors — simple kinds', () => {
	beforeEach(() => history().reset());

	it('add_node inverse removes the node by its assigned name', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		const action: Action = {
			kind: 'add_node',
			label: 'Add Oscillator',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { type: 'Oscillator', category: 'inputs', pos: [0, 0], uid: 'osc0' }
		};
		await graphExecutors['add_node'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls().some((c) => c.op === 'remove_node' && c.payload.node === 'osc0')).toBe(true);
		fc.emit({ event: 'node_removed', payload: { node: 'osc0', membership: null } });
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeUndefined();
	});

	it('add_node forward restores the original uid (member_uid) + name on redo', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('add_node', 'osc0');
		const action: Action = {
			kind: 'add_node',
			label: 'Add Oscillator',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { type: 'Oscillator', category: 'inputs', pos: [0, 0], uid: 'osc0', name: 'oscillator0' }
		};
		await graphExecutors['add_node'].forward(action, deps(fc, g));
		const call = fc.recordedCalls().find((c) => c.op === 'add_node');
		// redo must restore the SAME uid so captured links/panels reconnect
		expect(call?.payload.member_uid).toBe('osc0');
		expect(call?.payload.name).toBe('oscillator0');
	});

	it('rename_node forward/inverse send the uid as identity + the right name', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'rename_node',
			label: 'Rename',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { uid: 'osc0', oldName: 'oscillator0', newName: 'brain_wave' }
		};
		await graphExecutors['rename_node'].forward(action, deps(fc, g));
		expect(
			fc.recordedCalls().some((c) => c.op === 'rename_node' && c.payload.node === 'osc0' && c.payload.name === 'brain_wave')
		).toBe(true);
		await graphExecutors['rename_node'].inverse(action, deps(fc, g));
		expect(
			fc.recordedCalls().some((c) => c.op === 'rename_node' && c.payload.node === 'osc0' && c.payload.name === 'oscillator0')
		).toBe(true);
	});

	it('undo of removeNode re-adds the node with the same name and restores its links', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docAddNode(g, 'osc0');
		docAddNode(g, 'buffer0', 'Buffer');
		const link: LinkInfo = { node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' };
		docAddLink(g, link);

		// Build the remove_node action capturing pre-state from the store (the
		// recording wrapper does this in production; here we do it explicitly).
		const node = g.nodes.find((n) => n.name === 'osc0')!;
		const action: Action = {
			kind: 'remove_node',
			label: 'Delete osc0',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				uid: 'osc0',
				node: structuredClone(node),
				links: [link],
				membership: null,
				boundPanels: []
			}
		};

		await graphExecutors['remove_node'].forward(action, deps(fc, g));
		docRemoveNode(g, 'osc0');
		docClearLinks(g);
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeUndefined();
		expect(g.links).toHaveLength(0);

		await graphExecutors['remove_node'].inverse(action, deps(fc, g));
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node');
		expect(addCall).toBeDefined();
		expect(addCall!.payload.name).toBe('osc0'); // SAME display name
		expect(fc.recordedCalls().some((c) => c.op === 'add_link')).toBe(true);
		// simulate the backend echo
		docAddNode(g, 'osc0');
		docAddLink(g, link);
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeDefined();
		expect(g.links).toHaveLength(1);
	});

	it('instantiateNodes threads inst_id so paste lands inside the entered sub-patch', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('add_node', 'clone0');
		await g.instantiateNodes(
			[{ key: 'k0', type: 'Buffer', category: 'signal', pos: [0, 0], params: {} }],
			[],
			'subpatch0'
		);
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node');
		expect(addCall).toBeDefined();
		expect(addCall!.payload.inst_id).toBe('subpatch0'); // pasted node becomes a member
	});

	it('instantiateNodes with no inst_id pastes at root (undefined)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('add_node', 'clone0');
		await g.instantiateNodes([{ key: 'k0', type: 'Buffer', category: 'signal', pos: [0, 0], params: {} }]);
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node');
		expect(addCall!.payload.inst_id).toBeUndefined();
	});

	it('undo of deleting a sub-patch MEMBER restores it inside the sub-patch (not at root)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const node = nodeInfo('buffer0', 'Buffer');
		const action: Action = {
			kind: 'remove_node',
			label: 'Delete buffer0',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				uid: 'buffer0',
				node: structuredClone(node),
				links: [],
				membership: { instance: 'subpatch0', local_name: 'buffer0' },
				boundPanels: []
			}
		};
		void graphExecutors['remove_node'].inverse(action, deps(fc, g));
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node');
		expect(addCall).toBeDefined();
		// The member must be re-created back INSIDE its sub-patch (add_member_node path),
		// not orphaned at top level — so inst_id rides the captured membership.
		expect(addCall!.payload.inst_id).toBe('subpatch0');
		expect(addCall!.payload.member_uid).toBe('buffer0'); // same identity
	});

	it('add_link inverse removes the link and re-adds a displaced one', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const newLink: LinkInfo = { node_out: 'a', slot_out: 'o', node_in: 'b', slot_in: 'in' };
		const displaced: LinkInfo = { node_out: 'c', slot_out: 'o', node_in: 'b', slot_in: 'in' };
		const action: Action = {
			kind: 'add_link',
			label: 'Connect',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { link: newLink, displaced }
		};
		await graphExecutors['add_link'].inverse(action, deps(fc, g));
		const ops = fc.recordedCalls();
		expect(ops[0]).toEqual({ op: 'remove_link', payload: { ...newLink } });
		expect(ops[1]).toEqual({ op: 'add_link', payload: { ...displaced } });
	});

	it('remove_link inverse re-adds the link', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const link: LinkInfo = { node_out: 'a', slot_out: 'o', node_in: 'b', slot_in: 'in' };
		const action: Action = {
			kind: 'remove_link',
			label: 'Disconnect',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { link }
		};
		await graphExecutors['remove_link'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([{ op: 'add_link', payload: { ...link } }]);
	});

	it('set_expression inverse writes the prior expression binding to the doc (retired RPC)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		docAddNode(g, 'osc0');
		// A current binding the inverse (oldExpr = null) must clear.
		setParamExpr(g.doc, 'osc0', 'common', 'frequency', { source: 'nd("a")', enabled: true, triggers: false });
		const action: Action = {
			kind: 'set_expression',
			label: 'Set expression',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				node: 'osc0',
				group: 'common',
				name: 'frequency',
				oldExpr: { expression: null, enabled: false, triggers_process: false },
				newExpr: { expression: 'nd("a")', enabled: true, triggers_process: false }
			}
		};
		await graphExecutors['set_expression'].inverse(action, deps(fc, g));
		// The retired set_expression RPC is gone; the old (cleared) binding lands in the CRDT doc.
		expect(fc.recordedCalls().some((c) => c.op === 'set_expression')).toBe(false);
		expect(docParams(g.doc, 'osc0').common?.frequency?.expr).toBeUndefined();
	});

	it('update_param inverse writes the old value to the doc (leaf-write)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		docAddNode(g, 'osc0'); // the doc node the leaf-write targets
		const action: Action = {
			kind: 'update_param',
			label: 'Set freq',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { node: 'osc0', group: 'common', name: 'frequency', oldValue: 1, newValue: 5 }
		};
		await graphExecutors['update_param'].inverse(action, deps(fc, g));
		// The retired update_param RPC is gone; the old value lands in the CRDT doc instead.
		expect(fc.recordedCalls().some((c) => c.op === 'update_param')).toBe(false);
		expect(paramValue(g.doc, 'osc0', 'common', 'frequency')).toBe(1);
	});

	it('set_node_pos inverse writes the old position to the doc (retired RPC)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		docAddNode(g, 'osc0'); // the doc node the leaf-write targets
		const action: Action = {
			kind: 'set_node_pos',
			label: 'Move osc0',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { uid: 'osc0', oldPos: [10, 20], newPos: [99, 99] }
		};
		await graphExecutors['set_node_pos'].inverse(action, deps(fc, g));
		// The retired set_node_pos RPC is gone; the old position lands in the CRDT doc instead.
		expect(fc.recordedCalls().some((c) => c.op === 'set_node_pos')).toBe(false);
		expect(nodeView(g.doc, 'osc0')!.pos).toEqual([10, 20]);
	});
});

describe('graph executors — composite + sub-patch kinds', () => {
	beforeEach(() => history().reset());

	it('group_nodes: forward records the new instId, inverse expands it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('group_nodes', { inst_id: 'subpatch1' });
		const action: Action = {
			kind: 'group_nodes',
			label: 'Group',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { members: ['a', 'b'], instId: 'subpatch0' }
		};
		await graphExecutors['group_nodes'].forward(action, deps(fc, g));
		expect((action.payload as { instId: string }).instId).toBe('subpatch1'); // remapped
		await graphExecutors['group_nodes'].inverse(action, deps(fc, g));
		expect(
			fc.recordedCalls().some((c) => c.op === 'expand_instance' && c.payload.inst_id === 'subpatch1')
		).toBe(true);
	});

	it('group_nodes: forward passes the prior instId so a redo restores a stable id', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('group_nodes', { inst_id: 'subpatch0' });
		const action: Action = {
			kind: 'group_nodes',
			label: 'Group',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { members: ['a', 'b'], instId: 'subpatch0' } // already minted → this is a redo
		};
		await graphExecutors['group_nodes'].forward(action, deps(fc, g));
		const call = fc.recordedCalls().find((c) => c.op === 'group_nodes');
		// Redo must re-create the SAME instance id, else a stacked expand_instance(id) goes stale.
		expect(call?.payload.inst_id).toBe('subpatch0');
	});

	it('expand_instance: inverse passes the captured instId so the re-group restores it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('group_nodes', { inst_id: 'subpatch0' });
		const action: Action = {
			kind: 'expand_instance',
			label: 'Ungroup',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', restoredMembers: ['osc0', 'buffer0'], interface: {} }
		};
		await graphExecutors['expand_instance'].inverse(action, deps(fc, g));
		const call = fc.recordedCalls().find((c) => c.op === 'group_nodes');
		// Undo-of-expand must re-create the instance under its ORIGINAL id so the outer
		// group action's expand_instance(instId) still resolves on the next undo.
		expect(call?.payload.inst_id).toBe('subpatch0');
	});

	it('expand_instance: inverse re-groups the restored members', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('group_nodes', { inst_id: 'subpatch9' });
		const action: Action = {
			kind: 'expand_instance',
			label: 'Ungroup',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', restoredMembers: ['osc0', 'buffer0'], interface: {} }
		};
		await graphExecutors['expand_instance'].inverse(action, deps(fc, g));
		const call = fc.recordedCalls().find((c) => c.op === 'group_nodes');
		expect(call?.payload.members).toEqual(['osc0', 'buffer0']);
		expect((action.payload as { instId: string }).instId).toBe('subpatch9'); // remapped for redo
	});

	it('expand_instance: inverse restores the captured interface (unwired boundaries survive ungroup→undo)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('group_nodes', { inst_id: 'subpatch9' });
		// An UNWIRED In/Out port — it has no crossing link, so a derive-from-links re-group
		// would drop it; the inverse must replay the captured interface verbatim.
		const iface = { out0: { dir: 'out', dtype: 'ARRAY', inner_node: null, inner_slot: null, pos: [1, 2] } } as Record<
			string,
			import('$lib/api/control').SubPatchPort
		>;
		const action: Action = {
			kind: 'expand_instance',
			label: 'Ungroup',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', restoredMembers: ['osc0'], interface: iface }
		};
		await graphExecutors['expand_instance'].inverse(action, deps(fc, g));
		const call = fc.recordedCalls().find((c) => c.op === 'group_nodes');
		expect(call?.payload.interface).toEqual(iface);
	});

	it('add_boundary: forward records bnd_id, inverse removes it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('add_boundary', { bnd_id: 'in0' });
		const action: Action = {
			kind: 'add_boundary',
			label: 'Add input',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', bndId: '', dir: 'in', dtype: 'ARRAY', pos: [0, 0] }
		};
		await graphExecutors['add_boundary'].forward(action, deps(fc, g));
		expect((action.payload as { bndId: string }).bndId).toBe('in0');
		await graphExecutors['add_boundary'].inverse(action, deps(fc, g));
		expect(
			fc.recordedCalls().some((c) => c.op === 'remove_boundary' && c.payload.bnd_id === 'in0')
		).toBe(true);
	});

	it('wire_boundary: inverse restores the prior inner target', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'wire_boundary',
			label: 'Wire',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				instId: 'subpatch0',
				bndId: 'in0',
				oldInner: { node: null, slot: null },
				newInner: { node: 'osc0', slot: 'out' }
			}
		};
		await graphExecutors['wire_boundary'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([
			{ op: 'wire_boundary', payload: { inst_id: 'subpatch0', bnd_id: 'in0', inner_node: null, inner_slot: null } }
		]);
	});

	it('set_boundary_pos: inverse restores the old position', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'set_boundary_pos',
			label: 'Move pill',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', bndId: 'in0', oldPos: [1, 2], newPos: [9, 9] }
		};
		await graphExecutors['set_boundary_pos'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([
			{ op: 'set_boundary_pos', payload: { inst_id: 'subpatch0', bnd_id: 'in0', pos: [1, 2] } }
		]);
	});

	it('remove_boundary: inverse re-adds the port and rewires it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('add_boundary', { bnd_id: 'out1' });
		const action: Action = {
			kind: 'remove_boundary',
			label: 'Remove boundary',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				instId: 'subpatch0',
				bndId: 'out0',
				port: { dir: 'out', dtype: 'ARRAY', inner_node: 'osc0', inner_slot: 'out', pos: [3, 4] }
			}
		};
		await graphExecutors['remove_boundary'].inverse(action, deps(fc, g));
		const ops = fc.recordedCalls();
		expect(ops[0].op).toBe('add_boundary');
		expect(ops[1]).toEqual({
			op: 'wire_boundary',
			payload: { inst_id: 'subpatch0', bnd_id: 'out1', inner_node: 'osc0', inner_slot: 'out' }
		});
	});

	it('remove_boundary: inverse remaps bndId so a later redo removes the id that now exists', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		// The re-added boundary gets a FRESH id (the backend reclaims the lowest-unused
		// index, which differs from the original when a lower gap exists).
		fc.setCallResult('add_boundary', { bnd_id: 'out1' });
		const action: Action = {
			kind: 'remove_boundary',
			label: 'Remove boundary',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				instId: 'subpatch0',
				bndId: 'out0',
				port: { dir: 'out', dtype: 'ARRAY', inner_node: 'osc0', inner_slot: 'out', pos: [3, 4] }
			}
		};
		await graphExecutors['remove_boundary'].inverse(action, deps(fc, g));
		// The payload must track the re-added id, else redo removes a boundary that no
		// longer exists (KeyError) and the redo aborts.
		expect((action.payload as { bndId: string }).bndId).toBe('out1');

		await graphExecutors['remove_boundary'].forward(action, deps(fc, g));
		expect(
			fc.recordedCalls().some((c) => c.op === 'remove_boundary' && c.payload.bnd_id === 'out1')
		).toBe(true);
	});

	// (sharing removed: duplicate_shared / make_unique / re_share_instance executors are gone.)
});

describe('history.transaction — compound actions', () => {
	beforeEach(() => history().reset());

	it('wraps addNode + addLink into ONE undo entry; undo reverses both', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.setCallResult('add_node', 'osc0');
		fc.emit({ event: 'node_added', payload: nodeInfo('buffer0', 'Buffer') });

		await history().transaction('Add + wire', async () => {
			await g.addNode('Oscillator', 'inputs', [0, 0]);
			await g.addLink({ node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' });
		});
		expect(history().length).toBe(1); // one compound, not two
		expect(history().undoLabel).toBe('Add + wire');

		await history().undo();
		const ops = fc.recordedCalls().map((c) => c.op);
		// children reverse-ordered: remove the link first, then the node
		const iLink = ops.lastIndexOf('remove_link');
		const iNode = ops.lastIndexOf('remove_node');
		expect(iLink).toBeGreaterThanOrEqual(0);
		expect(iNode).toBeGreaterThan(iLink);
		expect(history().canUndo).toBe(false);
		expect(history().canRedo).toBe(true);
	});

	it('a single-child transaction records the child directly (no compound wrapper)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.setCallResult('add_node', 'osc0');
		await history().transaction('Add node', async () => {
			await g.addNode('Oscillator', 'inputs', [0, 0]);
		});
		expect(history().length).toBe(1);
		// unwrapped → the label is the child's, not the transaction's
		expect(history().undoLabel).toBe('Add Oscillator');
	});

	it('redo of a compound replays children in forward order', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.setCallResult('add_node', 'osc0');
		fc.emit({ event: 'node_added', payload: nodeInfo('buffer0', 'Buffer') });
		await history().transaction('Add + wire', async () => {
			await g.addNode('Oscillator', 'inputs', [0, 0]);
			await g.addLink({ node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' });
		});
		await history().undo();
		fc.recordedCalls().length = 0;
		await history().redo();
		const ops = fc.recordedCalls().map((c) => c.op);
		expect(ops.indexOf('add_node')).toBeGreaterThanOrEqual(0);
		expect(ops.indexOf('add_link')).toBeGreaterThan(ops.indexOf('add_node'));
	});
});

describe('graph store — recording wrappers + undo replay', () => {
	beforeEach(() => history().reset());

	it('suspend blocks recording during a 5-node batch', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		await history().suspend(async () => {
			for (let i = 0; i < 5; i++) await g.addNode('Oscillator', 'inputs', [0, 0]);
		});
		expect(history().canUndo).toBe(false);
	});

	it('removeNode records an action; history.undo() replays add_node + add_link', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		docAddNode(g, 'osc0');
		docAddNode(g, 'buffer0', 'Buffer');
		const link: LinkInfo = { node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' };
		docAddLink(g, link);

		await g.removeNode('osc0');
		expect(history().canUndo).toBe(true);
		docRemoveNode(g, 'osc0');
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeUndefined();

		await history().undo();
		const ops = fc.recordedCalls().map((c) => c.op);
		expect(ops).toContain('add_node');
		expect(ops).toContain('add_link');
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node')!;
		expect(addCall.payload.name).toBe('osc0');
		expect(history().canUndo).toBe(false);
		expect(history().canRedo).toBe(true);
	});

	it('undo of a multi-node delete restores every node before any inter-node link', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		history().configureDeps(() => deps(fc, g));
		docAddNode(g, 'osc0');
		docAddNode(g, 'buffer0', 'Buffer');
		const link: LinkInfo = { node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' };
		docAddLink(g, link);

		// Delete BOTH linked nodes as one batch, then mirror the backend teardown.
		await g.removeNodes(['osc0', 'buffer0']);
		docClearLinks(g);
		docRemoveNode(g, 'osc0');
		docRemoveNode(g, 'buffer0');

		fc.recordedCalls().length = 0; // inspect only the undo's RPCs
		await history().undo(); // ONE step restores the whole batch

		const ops = fc.recordedCalls().map((c) => c.op);
		expect(ops.filter((o) => o === 'add_node').length).toBe(2); // both nodes back in one step
		expect(ops).toContain('add_link'); // the inter-node link is restored
		// the link is re-added only AFTER both endpoints exist again
		expect(ops.indexOf('add_link')).toBeGreaterThan(ops.lastIndexOf('add_node'));
	});

	it('loadText records a load_patch entry; undo re-loads the prior YAML', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		workspace().reset();
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.setCallResult('serialize', { yaml: 'BEFORE_YAML' });

		await g.loadText('AFTER_YAML');
		expect(history().canUndo).toBe(true);
		expect(history().undoLabel).toBe('Load patch');

		fc.recordedCalls().length = 0; // clear to inspect the undo's calls
		await history().undo();
		const loadCall = fc.recordedCalls().find((c) => c.op === 'load_text');
		expect(loadCall?.payload.content).toBe('BEFORE_YAML'); // restored prior patch
	});

	it('redo of a load restores the layout captured after the load settled (#20)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const ws = workspace();

		// Build a recognisable 2-panel post-load layout, capture its shape, then
		// reset — so the measured phase starts from a clean history.
		ws.reset();
		ws.split(ws.activePanelId!, 'row', false, 0.5, 'viewer');
		const settledLayout = ws.serialize();
		expect(collectPanels(settledLayout.workspaces[0].root)).toHaveLength(2);
		ws.reset();
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: ws }));

		// Establish a backend session so the later graph_replaced echo (same id) is
		// an in-session load, not a fresh session (which would reset history).
		const hello = (instance_id: string, layout: unknown) =>
			fc.emit({
				event: 'hello',
				payload: { nodes: [], links: [], instances: {}, save_path: null, unsaved_changes: false, instance_id, layout } as never
			});
		hello('sess1', null);
		expect(collectPanels(ws.active.root)).toHaveLength(1); // pre-load baseline

		fc.setCallResult('serialize', { yaml: 'BEFORE_YAML' });
		await g.loadText('AFTER_YAML');

		// The load settles: the loaded patch carried the 2-panel layout. The
		// graph_replaced echo hydrates it; the store must capture it as afterLayout.
		fc.emit({
			event: 'graph_replaced',
			payload: { nodes: [], links: [], instances: {}, save_path: null, unsaved_changes: false, instance_id: 'sess1', layout: settledLayout } as never
		});
		expect(collectPanels(ws.active.root)).toHaveLength(2);

		// Undo → prior 1-panel layout; redo → must restore the captured 2-panel one
		// (a no-op back when afterLayout was hardcoded null).
		await history().undo();
		expect(collectPanels(ws.active.root)).toHaveLength(1);
		await history().redo();
		expect(collectPanels(ws.active.root)).toHaveLength(2);
	});

	it('a fresh backend session (changed instance_id) hard-resets the history', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		// record something
		docAddNode(g, 'osc0');
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		void g.removeNode('osc0');
		expect(history().canUndo).toBe(true);
		// a hello from a NEW backend instance
		fc.emit({
			event: 'hello',
			payload: {
				nodes: [],
				links: [],
				instances: {},
				save_path: null,
				unsaved_changes: false,
				instance_id: 'a-brand-new-session',
				layout: null
			} as never
		});
		expect(history().canUndo).toBe(false); // history dropped
	});

	it('restartNode respawns the node in place via restart_node, recording no history (#25)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		workspace().reset();
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		fc.emit({ event: 'node_added', payload: nodeInfo('vid0') });

		fc.recordedCalls().length = 0;
		await g.restartNode('vid0');

		const ops = fc.recordedCalls();
		// The backend restart_node respawns the process IN PLACE (keeps uid, name, params,
		// sub-patch membership and links), so the frontend issues a single restart_node RPC
		// — no remove+add+relink dance (which would re-home a member at ROOT / cascade shared).
		expect(ops).toHaveLength(1);
		expect(ops[0]).toMatchObject({ op: 'restart_node', payload: { node: 'vid0' } });
		expect(history().canUndo).toBe(false); // recovery op, not an edit
	});

	it('undo of removeNode restores panels that were bound to the node', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		const ws = workspace();
		ws.reset();
		// bind the (only) default panel to osc0
		const panelId = ws.activePanelId!;
		ws.setType(panelId, 'parameters');
		ws.linkNodeToPanel(panelId, 'osc0');
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: ws }));

		docAddNode(g, 'osc0');
		await g.removeNode('osc0');
		docRemoveNode(g, 'osc0');
		// the doc-removal reconcile (clearNodeRefs) clears the binding
		expect(ws.panelsBoundTo('osc0')).toHaveLength(0);

		await history().undo();
		docAddNode(g, 'osc0');
		expect(ws.panelsBoundTo('osc0').map((p) => p.panelId)).toContain(panelId);
	});
});

describe('sub-patch synth node identity (viewer re-instantiation bug)', () => {
	function withInstance(g: GraphStore, fc: FakeControl, iface: Record<string, unknown>): void {
		// The bridge ships a server-COMPUTED instance record (describe_instance): slots are
		// computed from wired boundaries, members are inverted to local -> {uid,is_instance}.
		const slots: { input: Record<string, string>; output: Record<string, string> } = {
			input: {},
			output: {}
		};
		for (const [bid, p] of Object.entries(iface as Record<string, { dir: string; dtype?: string; inner_node: string | null }>)) {
			if (p.inner_node == null) continue;
			(p.dir === 'in' ? slots.input : slots.output)[bid] = p.dtype ?? 'ARRAY';
		}
		fc.emit({
			event: 'hello',
			payload: {
				nodes: [],
				links: [],
				instances: {
					subpatch0: {
						uid: 'subpatch0',
						name: 'subpatch0',
						kind: 'subpatch',
						def_id: null,
						parent: null,
						interface: iface,
						pos: [0, 0],
						members: { oscillator0: { uid: 'm0', is_instance: false } },
						slots,
						siblings: [],
						error: null,
						viewers: {}
					}
				},
				save_path: null,
				unsaved_changes: false,
				instance_id: 's',
				layout: null
			} as never
		});
	}

	it('nodeById returns a STABLE reference for an unchanged sub-patch instance', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		withInstance(g, fc, { out0: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'out' } });
		const a = g.nodeById('subpatch0');
		const b = g.nodeById('subpatch0');
		expect(a).not.toBeNull();
		// Same object across calls — like a real node — so a flowNodes rebuild on a
		// mere selection change doesn't hand the viewer a fresh node and re-subscribe.
		expect(a).toBe(b);
	});

	it('returns an UPDATED node when the instance interface changes (no stale memo)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		withInstance(g, fc, { out0: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'out' } });
		const before = g.nodeById('subpatch0')!;
		expect(Object.keys(before.output_slots)).toEqual(['out0']);
		// Wire a second output boundary → the synth node must rebuild to expose it.
		withInstance(g, fc, {
			out0: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'out' },
			out1: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'aux' }
		});
		const after = g.nodeById('subpatch0')!;
		expect(Object.keys(after.output_slots).sort()).toEqual(['out0', 'out1']);
		expect(after).not.toBe(before); // content changed → fresh object
	});

	it('reflects a position move without changing identity (drag must not churn viewers)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		Y.transact(g.doc, () => {
			seedNode(nodesMap(g.doc), 'm0', 'Buffer', 'osc0m');
			seedInstance(instancesMap(g.doc), 'subpatch0', {
				name: 'subpatch0',
				members: { m0: false },
				stubs: [{ bnd_id: 'out0', dir: 'out', dtype: 'ARRAY', name: 'out0', inner_node: 'm0', inner_slot: 'out' }]
			});
		});
		const a = g.nodeById('subpatch0')!;
		docSetInstancePos(g, 'subpatch0', [120, 80]); // position is doc-owned now
		const b = g.nodeById('subpatch0')!;
		expect(b).toBe(a); // same identity across a move
		expect(b.pos).toEqual([120, 80]); // but position is current
	});
});

describe('subpatch_changed reconciles real nodes in place (no viewer churn)', () => {
	function subpatchSnapshot(
		nodes: NodeInstanceInfo[],
		instances: Record<string, unknown>
	): never {
		return {
			nodes,
			links: [],
			instances,
			save_path: null,
			unsaved_changes: false,
			instance_id: 's',
			layout: null
		} as never;
	}

	it('keeps a surviving node’s identity + live inline view across an unrelated doc change', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docAddNode(g, 'osc0');
		docAddNode(g, 'buffer0', 'Buffer');
		// User picks an inline viewer kind on osc0 — live state, not yet pushed/persisted.
		setInlineKind('osc0', 'out', 'line');
		const before = g.nodeById('osc0');
		expect(before).not.toBeNull();

		// An UNRELATED doc change (a third node lands) triggers a full doc reconcile; osc0
		// survives untouched, so its object identity and live inline view must be preserved.
		docAddNode(g, 'buffer0b', 'Buffer');

		// A wholesale array swap would hand back a fresh object (re-subscribing the viewer);
		// in-place reconcile keeps the same reference.
		expect(g.nodeById('osc0')).toBe(before);
		// And the blanket forgetInlineView would have wiped the live kind — it must survive.
		expect(rawInlineView('osc0', 'out').kind).toBe('line');
	});

	it('forgets a node that genuinely vanished from the doc', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		g.nodeTypes = catalog();
		docAddNode(g, 'osc0');
		docAddNode(g, 'gone0', 'Buffer');
		setInlineKind('gone0', 'out', 'line');
		docRemoveNode(g, 'gone0');
		expect(g.nodeById('gone0')).toBeNull();
		expect(rawInlineView('gone0', 'out').kind).toBeUndefined(); // its inline view is dropped
	});

	function instanceRecord(): Record<string, unknown> {
		return {
			uid: 'subpatch0',
			name: 'subpatch0',
			kind: 'subpatch',
			def_id: null,
			parent: null,
			interface: { out0: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'out' } },
			pos: [0, 0],
			members: { oscillator0: { uid: 'm0', is_instance: false } },
			slots: { input: {}, output: { out0: 'ARRAY' } },
			siblings: [],
			error: null,
			viewers: {}
		};
	}

	it('does NOT reset a surviving instance viewer’s collapsed state on resync', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		// Seed the instance (a fresh-session hello seeds its viewers once).
		fc.emit({
			event: 'hello',
			payload: {
				nodes: [],
				links: [],
				instances: { subpatch0: instanceRecord() },
				save_path: null,
				unsaved_changes: false,
				instance_id: 's',
				layout: null
			} as never
		});
		// User collapses the sub-patch's inline output viewer (live, un-pushed state).
		ui().setSlotExpanded('subpatch0', 'out0', false);
		expect(ui().isSlotExpanded('subpatch0', 'out0')).toBe(false);

		// A later structural edit (e.g. add a member elsewhere) fires subpatch_changed
		// with this instance SURVIVING and its viewers map still empty (push not landed).
		fc.emit({ event: 'subpatch_changed', payload: subpatchSnapshot([], { subpatch0: instanceRecord() }) });

		// Re-seeding a survivor would re-pop the viewer open → remount ViewerFeed →
		// churn (close+reopen) its data subscription. It must stay collapsed.
		expect(ui().isSlotExpanded('subpatch0', 'out0')).toBe(false);
	});
});
