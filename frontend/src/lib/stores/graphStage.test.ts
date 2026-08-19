import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { nodesMap } from '$lib/crdt/graphDoc';
import type { NodeTypeInfo } from '$lib/api/control';

/** The catalog (list_nodes) the manager provides — its presence flips the store to
 * doc-authoritative for node identity. */
function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'PSD',
			category: 'signal',
			doc: '',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: { data: 'ARRAY' },
			output_slots: { psd: 'ARRAY' },
			params: {}
		}
	];
}


describe('node lifecycle stage', () => {
	it('seeds stage from the doc and follows state_update', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'PSD', 'psd0', [0, 0]);
		// The node's identity is doc-owned; its lifecycle stage is event-sourced. A node the client
		// has only just learned of has reported nothing yet, and `creating` is exactly what that
		// means — the add answers before the node's thread has said anything.
		expect(g.nodeById('n1')).toBeDefined();
		expect(g.nodeById('n1')?.stage).toBe('creating');

		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, stage: 'setup' }
		});
		expect(g.nodeById('n1')?.stage).toBe('setup');

		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, stage: 'ready' }
		});
		expect(g.nodeById('n1')?.stage).toBe('ready');
	});

	it('state_update carries the error and applies it (a healthy respawn clears the stale chip)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'PSD', 'psd0', [0, 0]);

		// a setup() failure rides the idempotent state plane
		fc.emit({
			event: 'state_update',
			payload: {
				node: 'n1',
				params: {},
				stage: 'setup',
				error: 'RuntimeError: setup boom'
			}
		});
		expect(g.nodeById('n1')?.error).toContain('setup boom');

		// a healthy respawn's state push carries error=null -> the chip clears
		fc.emit({
			event: 'state_update',
			payload: { node: 'n1', params: {}, stage: 'ready', error: null }
		});
		expect(g.nodeById('n1')?.error).toBe(null);
	});

	it('node_stage error is terminal and carries the traceback', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		g.nodeTypes = catalog();
		d.node('n1', 'PSD', 'psd0', [0, 0]);

		fc.emit({
			event: 'node_stage',
			payload: { node: 'n1', stage: 'error', error: 'ModuleNotFoundError: torch' }
		});
		expect(g.nodeById('n1')?.stage).toBe('error');
		expect(g.nodeById('n1')?.error).toContain('ModuleNotFoundError');
	});
});
