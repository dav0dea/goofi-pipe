import { describe, it, expect } from 'vitest';
import { slotView } from './inlineView';
import { viewExecutors } from './viewExecutors';
import { GraphStore } from '$lib/stores/graph.svelte';
import { FakeControl } from '$lib/test/fakeControl';
import { seed } from '$lib/test/docSeed';
import type { Action, NavContext } from '$lib/stores/history.svelte';
import type { NodeTypeInfo } from '$lib/api/control';

const CTX: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

function catalog(): NodeTypeInfo[] {
	return [
		{
			type: 'Oscillator',
			category: 'inputs',
			doc: '',
			source: 'builtin',
			available: true,
			missing_deps: [],
			input_slots: {},
			output_slots: { out: 'ARRAY', sig: 'ARRAY' },
			params: {}
		} as unknown as NodeTypeInfo
	];
}

/** One node in the replica, plus the manager's half of the loop: `echo()` merges every viewer
 * patch `edit_node` was asked for, which is the only way a viewer edit becomes visible. */
function fixture() {
	const fc = new FakeControl();
	const g = new GraphStore(fc);
	g.nodeTypes = catalog();
	const d = seed(fc).node('osc0', 'Oscillator', 'osc0');
	return {
		view: (slot: string) => slotView(g.nodeById('osc0'), slot),
		deps: { control: {} as never, graph: g },
		echo: () => {
			const sent = fc.recordedCalls().filter((c) => c.op === 'edit_node' && c.payload.viewers);
			const whole: Record<string, object> = {};
			for (const c of sent)
				for (const [slot, v] of Object.entries(c.payload.viewers as Record<string, object>))
					whole[slot] = { ...whole[slot], ...v };
			const node = sent[sent.length - 1]!.payload.node as string;
			d.patch({ nodes: { [node]: { viewers: JSON.stringify(whole) } } });
		}
	};
}

describe('set_view executor — inline target', () => {
	it('inverse restores the prior viewer kind', async () => {
		const { view, deps, echo } = fixture();
		const action: Action = {
			kind: 'set_view',
			domain: 'view',
			label: 'Viewer → image',
			context: CTX,
			payload: {
				target: { kind: 'inline', node: 'osc0', slot: 'out' },
				before: { kind: 'line', settings: {} },
				after: { kind: 'image', settings: {} }
			}
		};
		await viewExecutors['set_view'].forward(action, deps);
		echo();
		expect(view('out').kind).toBe('image');
		await viewExecutors['set_view'].inverse(action, deps);
		echo();
		expect(view('out').kind).toBe('line');
	});

	it('inverse restores prior settings, and leaves the collapse the user has now', async () => {
		const { view, deps, echo } = fixture();
		// The slot is collapsed before the settings change — a replayed snapshot carries kind and
		// settings only, so nothing in an undo may re-open or shut a viewer.
		deps.graph.setSlotView('osc0', 'sig', { collapsed: true });
		echo();
		const action: Action = {
			kind: 'set_view',
			domain: 'view',
			label: 'Viewer setting',
			context: CTX,
			payload: {
				target: { kind: 'inline', node: 'osc0', slot: 'sig' },
				before: { kind: 'line', settings: { logY: false } },
				after: { kind: 'line', settings: { logY: true } }
			}
		};
		await viewExecutors['set_view'].forward(action, deps);
		echo();
		expect(view('sig').settings?.logY).toBe(true);
		await viewExecutors['set_view'].inverse(action, deps);
		echo();
		expect(view('sig').settings?.logY).toBe(false);
		expect(view('sig').collapsed, 'the collapse is untouched by either direction').toBe(true);
	});
});
