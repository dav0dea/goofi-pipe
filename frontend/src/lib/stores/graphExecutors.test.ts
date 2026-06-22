import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { graphExecutors } from './graphExecutors';
import { history, type Action, type ExecutorDeps, type NavContext } from './history.svelte';
import { collectPanels } from '$lib/workspace/model';
import type { NodeInstanceInfo, LinkInfo } from '$lib/api/control';

const EMPTY_CTX: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

function nodeInfo(name: string, type = 'Oscillator'): NodeInstanceInfo {
	return {
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
			payload: { type: 'Oscillator', category: 'inputs', pos: [0, 0], assignedName: 'osc0' }
		};
		await graphExecutors['add_node'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls().some((c) => c.op === 'remove_node' && c.payload.name === 'osc0')).toBe(true);
		fc.emit({ event: 'node_removed', payload: { name: 'osc0' } });
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeUndefined();
	});

	it('undo of removeNode re-adds the node with the same name and restores its links', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		fc.emit({ event: 'node_added', payload: nodeInfo('buffer0', 'Buffer') });
		const link: LinkInfo = { node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' };
		fc.emit({ event: 'link_added', payload: link });

		// Build the remove_node action capturing pre-state from the store (the
		// recording wrapper does this in production; here we do it explicitly).
		const node = g.nodes.find((n) => n.name === 'osc0')!;
		const action: Action = {
			kind: 'remove_node',
			label: 'Delete osc0',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				name: 'osc0',
				node: structuredClone(node),
				links: [link],
				membership: null,
				boundPanels: []
			}
		};

		await graphExecutors['remove_node'].forward(action, deps(fc, g));
		fc.emit({ event: 'node_removed', payload: { name: 'osc0' } });
		fc.emit({ event: 'link_removed', payload: link });
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeUndefined();
		expect(g.links).toHaveLength(0);

		await graphExecutors['remove_node'].inverse(action, deps(fc, g));
		const addCall = fc.recordedCalls().find((c) => c.op === 'add_node');
		expect(addCall).toBeDefined();
		expect(addCall!.payload.name).toBe('osc0'); // SAME display name
		expect(fc.recordedCalls().some((c) => c.op === 'add_link')).toBe(true);
		// simulate the backend echo
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		fc.emit({ event: 'link_added', payload: link });
		expect(g.nodes.find((n) => n.name === 'osc0')).toBeDefined();
		expect(g.links).toHaveLength(1);
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

	it('set_expression inverse restores the prior expression state', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'set_expression',
			label: 'Set expression',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: {
				node: 'osc0',
				group: 'common',
				name: 'frequency',
				oldExpr: { expression: null, enabled: false, triggers_process: false, autoeval: false },
				newExpr: { expression: 'nd("a")', enabled: true, triggers_process: false, autoeval: true }
			}
		};
		await graphExecutors['set_expression'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([
			{
				op: 'set_expression',
				payload: {
					node: 'osc0',
					group: 'common',
					name: 'frequency',
					expression: null,
					expression_enabled: false,
					expression_triggers_process: false,
					expression_autoeval: false
				}
			}
		]);
	});

	it('update_param inverse restores the old value', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'update_param',
			label: 'Set freq',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { node: 'osc0', group: 'common', name: 'frequency', oldValue: 1, newValue: 5 }
		};
		await graphExecutors['update_param'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([
			{ op: 'update_param', payload: { node: 'osc0', group: 'common', name: 'frequency', value: 1 } }
		]);
	});

	it('set_node_pos inverse restores the old position', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'set_node_pos',
			label: 'Move osc0',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { name: 'osc0', oldPos: [10, 20], newPos: [99, 99] }
		};
		await graphExecutors['set_node_pos'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([{ op: 'set_node_pos', payload: { name: 'osc0', pos: [10, 20] } }]);
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

	it('duplicate_shared: inverse removes the new sibling and re-uniques the source', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.setCallResult('duplicate_shared', { def_id: 'd', inst_id: 'subpatch2' });
		const action: Action = {
			kind: 'duplicate_shared',
			label: 'Duplicate shared',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', newInstId: '', wasUnique: true }
		};
		await graphExecutors['duplicate_shared'].forward(action, deps(fc, g));
		expect((action.payload as { newInstId: string }).newInstId).toBe('subpatch2');
		await graphExecutors['duplicate_shared'].inverse(action, deps(fc, g));
		const ops = fc.recordedCalls();
		expect(ops.some((c) => c.op === 'remove_node' && c.payload.name === 'subpatch2')).toBe(true);
		expect(ops.some((c) => c.op === 'make_unique' && c.payload.inst_id === 'subpatch0')).toBe(true);
	});

	it('make_unique: inverse re-shares when it was previously shared', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'make_unique',
			label: 'Make unique',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', defIdBefore: 'd' }
		};
		await graphExecutors['make_unique'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls().some((c) => c.op === 'duplicate_shared' && c.payload.inst_id === 'subpatch0')).toBe(
			true
		);
	});

	it('make_unique: inverse is a no-op when it was already unique', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const action: Action = {
			kind: 'make_unique',
			label: 'Make unique',
			domain: 'graph',
			context: EMPTY_CTX,
			payload: { instId: 'subpatch0', defIdBefore: null }
		};
		await graphExecutors['make_unique'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([]);
	});
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
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		fc.emit({ event: 'node_added', payload: nodeInfo('buffer0', 'Buffer') });
		const link: LinkInfo = { node_out: 'osc0', slot_out: 'out', node_in: 'buffer0', slot_in: 'in' };
		fc.emit({ event: 'link_added', payload: link });

		await g.removeNode('osc0');
		expect(history().canUndo).toBe(true);
		fc.emit({ event: 'node_removed', payload: { name: 'osc0' } });
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
		// record something
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
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

	it('restartNode respawns the node with its params + links and records no history (#25)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		workspace().reset();
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: workspace() }));

		const vid = nodeInfo('vid0');
		vid.input_slots = { in: 'ARRAY' };
		vid.params = { common: { fps: { value: 30 } } } as never;
		fc.emit({ event: 'node_added', payload: vid });
		fc.emit({ event: 'node_added', payload: nodeInfo('buf0', 'Buffer') });
		const lin: LinkInfo = { node_out: 'src0', slot_out: 'out', node_in: 'vid0', slot_in: 'in' };
		const lout: LinkInfo = { node_out: 'vid0', slot_out: 'out', node_in: 'buf0', slot_in: 'in' };
		fc.emit({ event: 'link_added', payload: lin });
		fc.emit({ event: 'link_added', payload: lout });

		fc.recordedCalls().length = 0;
		await g.restartNode('vid0');

		const ops = fc.recordedCalls();
		const iRemove = ops.findIndex((c) => c.op === 'remove_node' && c.payload.name === 'vid0');
		const iAdd = ops.findIndex((c) => c.op === 'add_node' && c.payload.name === 'vid0');
		expect(iRemove).toBeGreaterThanOrEqual(0);
		expect(iAdd).toBeGreaterThan(iRemove); // re-add AFTER remove, same name
		expect(ops[iAdd].payload.params).toEqual({ common: { fps: 30 } }); // params preserved
		expect(ops.filter((c) => c.op === 'add_link')).toHaveLength(2); // both links restored
		expect(history().canUndo).toBe(false); // recovery op, not an edit
	});

	it('undo of removeNode restores panels that were bound to the node', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const ws = workspace();
		ws.reset();
		// bind the (only) default panel to osc0
		const panelId = ws.activePanelId!;
		ws.setType(panelId, 'parameters');
		ws.linkNodeToPanel(panelId, 'osc0');
		history().reset();
		history().configureDeps(() => ({ control: fc, graph: g, workspace: ws }));

		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		await g.removeNode('osc0');
		fc.emit({ event: 'node_removed', payload: { name: 'osc0' } });
		// node_removed clears the binding
		expect(ws.panelsBoundTo('osc0')).toHaveLength(0);

		await history().undo();
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		expect(ws.panelsBoundTo('osc0').map((p) => p.panelId)).toContain(panelId);
	});
});
