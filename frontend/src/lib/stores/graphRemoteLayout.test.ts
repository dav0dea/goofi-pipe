import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { collectPanels } from '$lib/workspace/model';
import { asStateObject } from '$lib/workspace/panelState';
import { selection } from './selection.svelte';
import type { GraphSnapshot } from '$lib/api/control';

/**
 * The `layout` event: another client authored the arrangement, and this one has to converge.
 *
 * The manager broadcasts it to EVERY client including the author, because a broadcast channel has
 * no other shape. So the author filters its own echo out here, by the session id it already sends
 * on every request — without that, the author re-adopts its own blob, which re-seeds its panel ids
 * and drains its intent fold for nothing.
 */

function snapshot(over: Partial<GraphSnapshot> = {}): GraphSnapshot {
	return {
		runtime: {},
		save_path: null,
		unsaved_changes: false,
		instance_id: 'sess1',
		layout: null,
		...over
	} as never;
}

/** A store that has said hello, plus the peer blob a split would produce. */
function connected(): {
	fc: FakeControl;
	g: GraphStore;
	ws: ReturnType<typeof workspace>;
	remote: unknown;
} {
	const fc = new FakeControl();
	const g = new GraphStore(fc);
	fc.emit({ event: 'hello', payload: snapshot() });
	const ws = workspace();
	const base = ws.serialize();
	ws.split(ws.activePanelId!, 'row');
	const remote = ws.serialize();
	ws.hydrate(structuredClone(base));
	ws.takeLayoutIntent();
	ws.takeRemoteApplied();
	return { fc, g, ws, remote };
}

function panelCount(): number {
	return collectPanels(workspace().active.root).length;
}

describe('a peer’s layout event', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
		workspace().takeLayoutIntent();
		workspace().takeRemoteApplied();
	});

	it('converges this client on the arrangement another one authored', () => {
		const { fc, remote } = connected();
		fc.emit({ event: 'layout', payload: { layout: remote, session: 'another-tab' } });
		expect(panelCount(), 'the desktop split a panel; this client has it too').toBe(2);
	});

	it('merges it — a peer adding a panel does not move where WE are looking', () => {
		const { fc, ws, remote } = connected();
		const mine = ws.activePanelId!;
		ws.setPanelState(mine, { subpatchPath: '/inst0' }, 'navigation');
		selection().selectNodes(mine, ['osc0']);

		fc.emit({ event: 'layout', payload: { layout: remote, session: 'another-tab' } });

		const panel = collectPanels(ws.active.root).find((p) => p.id === mine);
		expect(asStateObject(panel?.state).subpatchPath, 'still three levels in').toBe('/inst0');
		expect([...selection().nodes(mine)], 'and still holding our selection').toEqual(['osc0']);
	});

	it('does not re-fit the editor — a peer’s panel is not a wholesale load', () => {
		const { fc, g, remote } = connected();
		const epoch = g.loadEpoch;
		fc.emit({ event: 'layout', payload: { layout: remote, session: 'another-tab' } });
		expect(g.loadEpoch, 'nothing was loaded; our viewport stays put').toBe(epoch);
	});

	it('ignores the echo of our OWN write', () => {
		const { fc, remote } = connected();
		fc.emit({ event: 'layout', payload: { layout: remote, session: fc.session } });
		expect(panelCount(), 'we already have our own arrangement').toBe(1);
		expect(workspace().takeRemoteApplied(), 'and nothing was applied to latch').toBe(false);
	});
});
