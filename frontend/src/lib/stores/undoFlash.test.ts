import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { flash } from './flash.svelte';
import { pulseRestored } from './undoFlash';
import type { NavContext } from './history.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

// uid is the identity; name is a separate, mutable display label. The fixture
// keeps them DISTINCT so a matcher that compares the echo's display name to a
// selection uid is caught.
function nodeInfo(uid: string, name: string): NodeInstanceInfo {
	return {
		uid,
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
	it('pulses a present node immediately and an absent one once its node_added echoes (matched by uid)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'node_added', payload: nodeInfo('uf_present', 'display-present') });

		// Selection sets are keyed by uid.
		const ctx: NavContext = {
			activeWorkspaceId: 'w',
			activePanelId: 'p',
			enteredPath: {},
			selection: { p: { nodes: ['uf_present', 'uf_absent'], edges: [] } }
		};
		pulseRestored(ctx, { control: fc, graph: g, workspace: workspace() });

		expect(flash().active('uf_present')).toBe(true);
		expect(flash().active('uf_absent')).toBe(false); // not in the graph yet

		// The re-created node echoes with its uid AND a display name that differs
		// from the uid — the flash must key on the uid, not the display name.
		fc.emit({ event: 'node_added', payload: nodeInfo('uf_absent', 'display-absent') });
		await Promise.resolve();
		await Promise.resolve();
		expect(flash().active('uf_absent')).toBe(true);
	});
});
