import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { graphExecutors } from './graphExecutors';
import { history, type Action, type ExecutorDeps, type NavContext } from './history.svelte';
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
});
