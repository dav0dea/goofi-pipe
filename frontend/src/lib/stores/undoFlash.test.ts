import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { flash } from './flash.svelte';
import { pulseRestored } from './undoFlash';
import type { NavContext } from './history.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

function nodeInfo(name: string): NodeInstanceInfo {
	return {
		uid: name,
		name,
		type: 'Oscillator',
		category: 'inputs',
		doc: '',
		input_slots: {},
		output_slots: { out: 'ARRAY' },
		params: {},
		pos: [0, 0],
		viewers: {},
		membership: null,
		error: null
	};
}

describe('pulseRestored', () => {
	it('pulses a present node immediately and an absent one once its node_added echoes', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeInfo('uf_present') });

		const ctx: NavContext = {
			activeWorkspaceId: 'w',
			activePanelId: 'p',
			enteredPath: {},
			selection: { p: { nodes: ['uf_present', 'uf_absent'], edges: [] } }
		};
		pulseRestored(ctx, { control: fc, graph: g, workspace: workspace() });

		expect(flash().active('uf_present')).toBe(true);
		expect(flash().active('uf_absent')).toBe(false); // not in the graph yet

		fc.emit({ event: 'node_added', payload: nodeInfo('uf_absent') });
		await Promise.resolve();
		await Promise.resolve();
		expect(flash().active('uf_absent')).toBe(true);
	});
});
