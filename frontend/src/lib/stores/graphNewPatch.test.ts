import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';

/**
 * `New` is a manager transaction, and the client's whole job is to *not* get in its way.
 *
 * Two things could go wrong, and both are invisible until much later. Recording a history entry
 * would desync the two undo stacks 1:1 — the manager CLEARS its history inside the `new` arm, so a
 * client entry would pop against nothing. And writing `savePath = null` here would be a second
 * writer on manager-owned state (C38), which is the exact shape the archive work removed from
 * `save`. The null path rides the `graph_replaced` snapshot instead, because `new` — unlike `load`
 * — emits no `save_path_changed`: the manager only announces a path it HAS.
 */
describe('GraphStore.newPatch — the reset door', () => {
	beforeEach(() => history().reset());

	it('sends `new` with an empty payload and records no undo step', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: snapshot('/patches/a.gfi') });
		expect(g.savePath).toBe('/patches/a.gfi');

		await g.newPatch();

		// `list_nodes` is the boot catalog fetch a hello with no `node_types` triggers — not ours.
		const mine = fc.recordedCalls().filter((c) => c.op !== 'list_nodes');
		expect(mine).toEqual([{ op: 'new', payload: {} }]);
		expect(history().canUndo, 'a New is not undoable — the manager dropped its history').toBe(
			false
		);
	});

	it('takes the patch name down off the snapshot, with no `save_path_changed` to help', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		fc.emit({ event: 'hello', payload: snapshot('/patches/a.gfi') });

		await g.newPatch();
		fc.emit({ event: 'graph_replaced', payload: snapshot(null) });

		expect(g.savePath).toBe(null);
		expect(g.unsavedChanges).toBe(false);
	});

	/**
	 * …and takes the panels down with it. A `graph_replaced` is never a reconnect — it is always a
	 * wholesale patch swap — so a null `layout` on one means "this patch has no arrangement", not
	 * "keep yours". The ambiguity it used to be read with belongs to `hello` alone.
	 *
	 * Left standing it is worse than a stale view: AppShell pushes `ws.serialize()` on the next
	 * split or tab switch, and `set_layout` persists regardless of intent — so the previous patch's
	 * arrangement becomes the new one's stored layout and rides into its `.gfi`. A layout-less
	 * `.gfi` (the engine writes one, and the Load button reaches it today) lands here too.
	 */
	// (That a New patch opens on the DEFAULT arrangement is the manager's now — `new` reloads an
	// empty document, which restores the default layout with it. Pinned over the wire by
	// `a_new_patch_is_empty_clean_and_unnamed`.)
});

function snapshot(savePath: string | null) {
	return {
		runtime: {},
		save_path: savePath,
		unsaved_changes: false,
		instance_id: 'sess1',
		viewpoint: null
	} as never;
}
