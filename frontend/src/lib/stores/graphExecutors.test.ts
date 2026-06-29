import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { graphExecutors } from './graphExecutors';
import { history, type Action, type ExecutorDeps, type NavContext } from './history.svelte';
import { collectPanels } from '$lib/workspace/model';
import { setInlineKind, rawInlineView } from '$lib/viewers/inlineView.svelte';
import { ui } from '$lib/stores/ui.svelte';
import type { NodeInstanceInfo, LinkInfo } from '$lib/api/control';

const EMPTY_CTX: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

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
		fc.emit({ event: 'node_removed', payload: { node: 'osc0' } });
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
				uid: 'osc0',
				node: structuredClone(node),
				links: [link],
				membership: null,
				boundPanels: []
			}
		};

		await graphExecutors['remove_node'].forward(action, deps(fc, g));
		fc.emit({ event: 'node_removed', payload: { node: 'osc0' } });
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
			payload: { uid: 'osc0', oldPos: [10, 20], newPos: [99, 99] }
		};
		await graphExecutors['set_node_pos'].inverse(action, deps(fc, g));
		expect(fc.recordedCalls()).toEqual([{ op: 'set_node_pos', payload: { node: 'osc0', pos: [10, 20] } }]);
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
		expect(ops.some((c) => c.op === 'remove_node' && c.payload.node === 'subpatch2')).toBe(true);
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
		fc.emit({ event: 'node_removed', payload: { node: 'osc0' } });
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
		const iRemove = ops.findIndex((c) => c.op === 'remove_node' && c.payload.node === 'vid0');
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
		fc.emit({ event: 'node_removed', payload: { node: 'osc0' } });
		// node_removed clears the binding
		expect(ws.panelsBoundTo('osc0')).toHaveLength(0);

		await history().undo();
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
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
						viewers: {},
						member_count: 1
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

	it('a live error event for an instance uid lights up the collapsed group node', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		withInstance(g, fc, { out0: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'out' } });
		expect(g.nodeById('subpatch0')!.error).toBeNull();
		// The backend fires `error` per ancestor instance (keyed by the instance uid)
		// when a deep member errors; the mirror updates inst.error -> the synth border.
		fc.emit({ event: 'error', payload: { node: 'subpatch0', error: 'deep boom' } });
		expect(g.instances['subpatch0'].error).toBe('deep boom');
		expect(g.nodeById('subpatch0')!.error).toBe('deep boom');
		// clearing propagates too
		fc.emit({ event: 'error', payload: { node: 'subpatch0', error: null } });
		expect(g.nodeById('subpatch0')!.error).toBeNull();
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
		withInstance(g, fc, { out0: { dir: 'out', dtype: 'ARRAY', inner_node: 'm0', inner_slot: 'out' } });
		const a = g.nodeById('subpatch0')!;
		fc.emit({ event: 'node_moved', payload: { node: 'subpatch0', pos: [120, 80] } });
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

	it('keeps a surviving node’s identity + live inline view across a group op', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		fc.emit({ event: 'node_added', payload: nodeInfo('buffer0', 'Buffer') });
		// User picks an inline viewer kind on osc0 — live state, not yet pushed/persisted.
		setInlineKind('osc0', 'out', 'line');
		const before = g.nodeById('osc0');
		expect(before).not.toBeNull();

		// A group op reparents buffer0 into a new instance; BOTH real nodes survive with
		// their uids. Only this whole-snapshot event fires (no per-node add/remove).
		fc.emit({
			event: 'subpatch_changed',
			payload: subpatchSnapshot([nodeInfo('osc0'), nodeInfo('buffer0', 'Buffer')], {
				subpatch0: {
					uid: 'subpatch0',
					name: 'subpatch0',
					kind: 'subpatch',
					def_id: null,
					parent: null,
					interface: {},
					pos: [0, 0],
					members: { buffer0: { uid: 'buffer0', is_instance: false } },
					slots: { input: {}, output: {} },
					siblings: [],
					error: null,
					viewers: {},
					member_count: 1
				}
			})
		});

		// A wholesale array swap would hand back a fresh object (re-subscribing the viewer);
		// in-place reconcile keeps the same reference.
		expect(g.nodeById('osc0')).toBe(before);
		// And the blanket forgetInlineView would have wiped the live kind — it must survive.
		expect(rawInlineView('osc0', 'out').kind).toBe('line');
	});

	it('forgets a node that genuinely vanished from the snapshot', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeInfo('osc0') });
		fc.emit({ event: 'node_added', payload: nodeInfo('gone0', 'Buffer') });
		setInlineKind('gone0', 'out', 'line');
		fc.emit({ event: 'subpatch_changed', payload: subpatchSnapshot([nodeInfo('osc0')], {}) });
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
			viewers: {},
			member_count: 1
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
