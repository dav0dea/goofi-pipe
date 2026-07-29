import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import type { GraphSnapshot } from '$lib/api/control';

/**
 * A patch that has just been loaded MATCHES DISK, and must push its layout as navigation.
 *
 * Getting there is a race the store cannot see: a load's CRDT delta and its queued JSON events
 * arrive independently, and when the delta wins, every uid in the old graph has already vanished
 * — `Graph::clear` does not reset `next_uid`, so nothing is reused — and `_reconcileNodes` calls
 * `clearNodeRefs` for each panel still bound to one. That marks *authored*, correctly: it is
 * normally a node DELETE that empties a panel, and a delete does change the patch.
 *
 * `hydrate`/`reset` drain that fold, so the case where the snapshot carries a layout was already
 * covered (layoutIntent.test.ts). What was not: a load where NEITHER runs. A layout-less `.gfi` is
 * explicitly supported by the engine, and on a same-session `graph_replaced` a null layout takes
 * neither branch — so the mark survived, and the next debounced push dirtied a patch identical to
 * the file the user had just opened.
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

/** A store that has said hello, with one panel bound to a node — the state a load arrives into. */
function loaded(): { fc: FakeControl; ws: ReturnType<typeof workspace> } {
	const fc = new FakeControl();
	const g = new GraphStore(fc);
	void g;
	fc.emit({ event: 'hello', payload: snapshot() });
	const ws = workspace();
	ws.linkNodeToPanel(ws.activePanelId!, 'osc0');
	ws.takeLayoutIntent(); // the bind itself IS authoring; isolate what the load does
	return { fc, ws };
}

describe('a wholesale load leaves nothing authored behind', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
		workspace().takeLayoutIntent();
	});

	it('drops the delete-storm mark when the loaded patch carries no layout', () => {
		const { fc, ws } = loaded();
		ws.clearNodeRefs('osc0'); // the CRDT delta beat the JSON event
		fc.emit({ event: 'graph_replaced', payload: snapshot({ layout: null }) });
		expect(ws.takeLayoutIntent(), 'a freshly loaded patch matches disk').toBe('navigation');
	});

	it('drops it when the loaded layout is malformed and hydrate refuses it', () => {
		const { fc, ws } = loaded();
		ws.clearNodeRefs('osc0');
		// `hydrate` bails before `_replaced()` on anything `isValidState` rejects.
		fc.emit({ event: 'graph_replaced', payload: snapshot({ layout: { workspaces: [] } as never }) });
		expect(ws.takeLayoutIntent(), 'the load still happened').toBe('navigation');
	});

	/* The other side of the same line, stated so the fix cannot be widened into swallowing a real
	   edit: a TRANSIENT reconnect to the same backend is not a wholesale load. It re-sends hello
	   with the layout it already has, the client keeps its state and its history — and an edit the
	   user made while the socket was down is still an edit. */
	it('keeps an authored mark across a same-session reconnect', () => {
		const { fc, ws } = loaded();
		ws.split(ws.activePanelId!, 'row');
		fc.emit({ event: 'hello', payload: snapshot() }); // same instance_id ⇒ a reconnect
		expect(ws.takeLayoutIntent(), 'a reconnect is not a load').toBe('authored');
	});
});
