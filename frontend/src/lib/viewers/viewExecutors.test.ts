import { describe, it, expect, beforeEach } from 'vitest';
import { setInlineKind, rawInlineView } from './inlineView.svelte';
import { viewExecutors } from './viewExecutors';
import { workspace } from '$lib/workspace/workspace.svelte';
import { findPanel } from '$lib/workspace/model';
import { GraphStore } from '$lib/stores/graph.svelte';
import { FakeControl } from '$lib/test/fakeControl';
import type { Action, NavContext } from '$lib/stores/history.svelte';

const CTX: NavContext = { activeWorkspaceId: 'w', activePanelId: null, enteredPath: {}, selection: {} };

function deps() {
	return { control: {} as never, graph: new GraphStore(new FakeControl()), workspace: workspace() };
}

describe('set_view executor — inline target', () => {
	it('inverse restores the prior viewer kind', async () => {
		setInlineKind('osc0', 'out', 'image');
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
		await viewExecutors['set_view'].inverse(action, deps());
		expect(rawInlineView('osc0', 'out').kind).toBe('line');
		await viewExecutors['set_view'].forward(action, deps());
		expect(rawInlineView('osc0', 'out').kind).toBe('image');
	});

	it('inverse restores prior settings', async () => {
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
		await viewExecutors['set_view'].forward(action, deps());
		expect(rawInlineView('osc0', 'sig').settings.logY).toBe(true);
		await viewExecutors['set_view'].inverse(action, deps());
		expect(rawInlineView('osc0', 'sig').settings.logY).toBe(false);
	});
});

describe('set_view executor — panel target', () => {
	beforeEach(() => workspace().reset());

	it('inverse restores the panel viewer kind, preserving node/slot', async () => {
		const ws = workspace();
		const pid = ws.activePanelId!;
		ws.setPanelState(pid, { node: 'osc0', slot: 'out', kind: 'image', settings: {} });
		const action: Action = {
			kind: 'set_view',
			domain: 'view',
			label: 'Panel viewer → image',
			context: CTX,
			payload: {
				target: { kind: 'panel', panelId: pid },
				before: { kind: 'line', settings: {} },
				after: { kind: 'image', settings: {} }
			}
		};
		await viewExecutors['set_view'].inverse(action, deps());
		const state = findPanel(ws.active.root, pid)?.state as Record<string, unknown>;
		expect(state.kind).toBe('line');
		expect(state.node).toBe('osc0'); // unrelated keys preserved
		expect(state.slot).toBe('out');
	});
});
