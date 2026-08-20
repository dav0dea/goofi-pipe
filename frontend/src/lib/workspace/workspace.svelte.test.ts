/**
 * The workspace store as a REPLICA: every frozen gesture leaves as a layout command, nothing edits
 * the arrangement locally, and the viewpoint never leaves as one.
 *
 * These are the tests that pin the cutover. Before it, each gesture rewrote a client-owned tree and
 * `set_layout` pushed the whole blob back debounced; now the tree is the manager's and the gesture
 * is an op, so what a gesture SENDS is the behaviour worth pinning.
 */
import { describe, expect, it, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { workspace } from './workspace.svelte';
import { history } from '$lib/stores/history.svelte';
import type { Arrangement } from './arrangement';

/** The manager's default arrangement, plus a row split on demand. */
function oneP(): Arrangement {
	return {
		'page-1': { kind: 'page', order: 0, name: 'Layout' },
		'panel-2': { kind: 'panel', order: 0, parent: 'page-1', size: 1, panel_type: 'node-editor' }
	};
}
function split(): Arrangement {
	return {
		'page-1': { kind: 'page', order: 0, name: 'Layout' },
		'split-4': { kind: 'split', order: 0, parent: 'page-1', size: 1, axis: 'row' },
		'panel-2': { kind: 'panel', order: 0, parent: 'split-4', size: 0.6, panel_type: 'node-editor' },
		'panel-3': { kind: 'panel', order: 1, parent: 'split-4', size: 0.4, panel_type: 'console' }
	};
}

let fc: FakeControl;

function boot(arr: Arrangement = oneP()): ReturnType<typeof workspace> {
	fc = new FakeControl();
	const ws = workspace();
	ws.configureControl(() => fc);
	// The store is a module singleton, so a drag a previous test armed would still be drawing:
	// committing a split that is not the drag's discards it, which is what an abandoned one does.
	ws.commitResize('#none');
	ws.syncFromDoc(arr);
	return ws;
}

/** Drain every pending microtask — a command's reply and whatever its `.then` does with it. */
const settle = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/** The ops sent since boot, as `[op, payload]` pairs. */
function sent(): Array<[string, Record<string, unknown>]> {
	return fc.recordedCalls().map((c) => [c.op, c.payload]);
}

beforeEach(() => history().reset());

describe('a frozen gesture is a layout command', () => {
	it('rebuilds the tree the manager holds, and holds none of its own', () => {
		const ws = boot(split());
		expect(ws.state.workspaces).toHaveLength(1);
		const root = ws.active.root;
		expect(root.kind).toBe('split');
		if (root.kind !== 'split') return;
		expect(root.children.map((c) => c.id)).toEqual(['panel-2', 'panel-3']);
	});

	it('splits through page_split_panel, carrying the side the drag went', async () => {
		const ws = boot();
		ws.split('panel-2', 'column', true, 0.25);
		await Promise.resolve();
		expect(sent()).toEqual([
			[
				'page_split_panel',
				{ page: 'Layout', panel: 'panel-2', direction: 'column', place_before: true, ratio: 0.25 }
			]
		]);
	});

	it('closes, retypes and re-binds through the page ops', async () => {
		const ws = boot(split());
		ws.close('panel-3');
		ws.setType('panel-3', 'viewer');
		ws.linkNodeToPanel('panel-3', 'a1b2');
		await Promise.resolve();
		expect(sent().map(([op]) => op)).toEqual([
			'page_remove_panel',
			'page_set_panel',
			'page_set_panel'
		]);
		expect(sent()[2][1]).toEqual({
			page: 'Layout',
			panel: 'panel-3',
			state: { node: 'a1b2', slot: null }
		});
	});

	it('settles the slot as it binds, so a picked node is never shown through the old node’s slot', async () => {
		// One write, so it is one undo step — and the SAME write for both doors onto the binding (a
		// node dragged in, a node picked from the bar's dropdown), which is what keeps them honest.
		const bound = split();
		bound['panel-3'].state = '{"node":"a1b2","slot":"spectrum"}';
		const ws = boot(bound);
		ws.linkNodeToPanel('panel-3', 'c3d4');
		await Promise.resolve();
		expect(sent()[0][1].state).toEqual({ node: 'c3d4', slot: null });
	});

	it('names only the key a panel write changes, never the bag it read', async () => {
		// A read-modify-write of the whole bag loses whatever a write still in flight put there:
		// `page_set_panel` merges, so the client sends the DELTA and the two orders cannot fight.
		const bound = split();
		bound['panel-3'].state = '{"node":"a1b2","kind":"line"}';
		const ws = boot(bound);
		ws.setPanelSlot('panel-3', 'out');
		ws.unlinkNodeFromPanel('panel-3');
		await Promise.resolve();
		expect(sent().map(([, p]) => p.state)).toEqual([{ slot: 'out' }, { node: null }]);
	});

	it('writes a panel that lives on a page in the background', async () => {
		// The replica is page-agnostic, and the façade an agent drives addresses any panel. Scoping
		// the lookup to the page in FRONT made a write to any other one silently do nothing.
		const two = split();
		two['page-7'] = { kind: 'page', order: 1, name: 'Second' };
		two['panel-8'] = { kind: 'panel', order: 0, parent: 'page-7', size: 1, panel_type: 'viewer' };
		const ws = boot(two);
		ws.linkNodeToPanel('panel-8', 'a1b2');
		await Promise.resolve();
		expect(sent()).toEqual([
			['page_set_panel', { page: 'Second', panel: 'panel-8', state: { node: 'a1b2', slot: null } }]
		]);
	});

	it('leaves the focus on the panel it dropped, not on the page’s first', async () => {
		const ws = boot(split());
		ws.setActive('panel-2');
		ws.dragging = { kind: 'panel', workspaceId: 'page-1', panelId: 'panel-3' };
		ws.dropOnPanel('panel-2', 'column', false);
		await settle();
		expect(ws.activePanelId, 'the panel the user just moved is the one they are in').toBe('panel-3');
	});

	it('brings a fresh tab forward off the ids the manager minted, once they arrive', async () => {
		const ws = boot();
		fc.setCallResult('session_add_page', { page: 'page-3', panel: 'panel-4' });
		ws.addTab();
		await settle();
		expect(ws.state.activeWorkspaceId, 'not before the page exists to draw').toBe('page-1');
		const two = oneP();
		two['page-3'] = { kind: 'page', order: 1, name: 'Layout 2' };
		two['panel-4'] = { kind: 'panel', order: 0, parent: 'page-3', size: 1, panel_type: 'node-editor' };
		ws.syncFromDoc(two);
		expect(ws.state.activeWorkspaceId).toBe('page-3');
		expect(ws.activePanelId).toBe('panel-4');
	});

	it('moves a dragged panel with ONE op, so the drop is one undo step', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'panel', workspaceId: 'page-1', panelId: 'panel-3' };
		ws.dropOnPanel('panel-2', 'column', false);
		await Promise.resolve();
		expect(sent()).toEqual([
			[
				'page_insert_at_panel',
				{
					page: 'Layout',
					subtree: 'panel-3',
					target: 'panel-2',
					direction: 'column',
					place_before: false,
					ratio: 0.5
				}
			]
		]);
		expect(ws.dragging, 'the drag is spent either way').toBeNull();
	});

	it('carries a whole tab’s subtree when the tab itself is dragged', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'tab', workspaceId: 'page-1' };
		ws.dropOnPanel('panel-2', 'row', false);
		await Promise.resolve();
		expect(sent()[0][1], 'a tab drag names the page’s root, not a panel').toMatchObject({
			subtree: 'split-4'
		});
	});

	it('tears a panel onto the tab bar as one page built around it', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'panel', workspaceId: 'page-1', panelId: 'panel-3' };
		ws.dropPanelOnTabBar(0);
		await Promise.resolve();
		expect(sent()).toEqual([
			['session_add_page', { name: 'Layout 2', index: 0, subtree: 'panel-3' }]
		]);
	});

	it('closing the tab in front moves to its NEIGHBOUR, not to the strip’s first', async () => {
		const three: Arrangement = {
			'page-1': { kind: 'page', order: 0, name: 'Layout' },
			'panel-2': { kind: 'panel', order: 0, parent: 'page-1', size: 1, panel_type: 'node-editor' },
			'page-3': { kind: 'page', order: 1, name: 'Two' },
			'panel-4': { kind: 'panel', order: 0, parent: 'page-3', size: 1, panel_type: 'console' },
			'page-5': { kind: 'page', order: 2, name: 'Three' },
			'panel-6': { kind: 'panel', order: 0, parent: 'page-5', size: 1, panel_type: 'console' }
		};
		const ws = boot(three);
		ws.selectTab('page-5');
		ws.closeTab('page-5');
		expect(ws.state.activeWorkspaceId, 'the neighbour, before the delta even lands').toBe('page-3');
		await Promise.resolve();
		expect(sent()).toEqual([['session_remove_page', { name: 'Three' }]]);
	});

	it('claims a fresh page name per tap, so a repeated gesture is not five refusals', async () => {
		const ws = boot();
		ws.addTab();
		ws.addTab();
		ws.addTab();
		await Promise.resolve();
		await Promise.resolve();
		const names = sent().map(([, p]) => p.name as string);
		expect(names, 'three taps, three requests').toHaveLength(3);
		expect(
			new Set(names).size,
			'each asks for a name the last one did not — the replica cannot have caught up between taps'
		).toBe(3);
		expect(names, 'and none of them is the name the page already has').not.toContain('Layout');
	});

	it('addresses a tab by the name the manager holds', async () => {
		const ws = boot(split());
		ws.renameTab('page-1', 'Signals');
		ws.closeTab('page-1');
		ws.reorderTab(0, 0);
		await Promise.resolve();
		expect(sent()).toEqual([
			['session_rename_page', { from: 'Layout', to: 'Signals' }],
			['session_remove_page', { name: 'Layout' }],
			['session_reorder_page', { name: 'Layout', to_index: 0 }]
		]);
	});
});

describe('a resize drag draws locally and commits once', () => {
	it('sends nothing while the pointer moves, and one op when it lifts', async () => {
		const ws = boot(split());
		ws.resize('split-4', 0, 0.1);
		ws.resize('split-4', 0, 0.05);
		await Promise.resolve();
		expect(sent(), 'a command per pointermove is exactly what this replaces').toEqual([]);

		const root = ws.active.root;
		if (root.kind !== 'split') throw new Error('expected a split');
		expect(root.sizes[0], 'but the seam moved on screen').toBeCloseTo(0.75, 6);

		ws.commitResize('split-4');
		await Promise.resolve();
		expect(sent()).toHaveLength(1);
		const [op, payload] = sent()[0];
		expect(op).toBe('page_resize_split');
		expect(payload).toMatchObject({ page: 'Layout', split: 'split-4' });
		const fractions = payload.fractions as number[];
		expect(fractions[0]).toBeCloseTo(0.75, 6);
		expect(fractions[1]).toBeCloseTo(0.25, 6);
	});

	it('keeps drawing the shares it drew until the manager’s answer lands', async () => {
		const ws = boot(split());
		ws.resize('split-4', 0, 0.1);
		ws.commitResize('split-4');
		await Promise.resolve();
		const mid = ws.active.root;
		if (mid.kind !== 'split') throw new Error('expected a split');
		expect(mid.sizes[0], 'no snap-back between the reply and the delta').toBeCloseTo(0.7, 6);

		const answered = split();
		answered['panel-2'].size = 0.7;
		answered['panel-3'].size = 0.3;
		ws.syncFromDoc(answered);
		const after = ws.active.root;
		if (after.kind !== 'split') throw new Error('expected a split');
		expect(after.sizes).toEqual([0.7, 0.3]);
	});

	it('commits nothing when the drag drew nothing', async () => {
		const ws = boot(split());
		ws.commitResize('split-4');
		await Promise.resolve();
		expect(sent()).toEqual([]);
	});

	it('commits a second drag that lands back on the shares the REPLICA still shows', async () => {
		// The "nothing changed" short-circuit has to compare against what was last SENT, not against
		// the replica — the replica is the arrangement from before the previous commit, so a drag
		// returning the split to those shares looks like a no-op and is dropped on the floor. Halves
		// and quarters, because floating-point inexactness is what hides this in a fixture.
		const even = split();
		even['panel-2'].size = 0.5;
		even['panel-3'].size = 0.5;
		const ws = boot(even);
		ws.resize('split-4', 0, 0.25);
		ws.commitResize('split-4');
		await Promise.resolve();
		ws.resize('split-4', 0, -0.25);
		ws.commitResize('split-4');
		await Promise.resolve();
		expect(sent(), 'the drag back is a change the user made and asked for').toHaveLength(2);
		expect(sent()[1][1].fractions).toEqual([0.5, 0.5]);
	});

	it('does not retire the override under a finger still on the seam', () => {
		// The previous commit's delta lands while the NEXT drag is live. Retiring the override then
		// jumps the seam out from under the pointer and the gesture continues from the jump.
		const ws = boot(split());
		ws.resize('split-4', 0, 0.1);
		ws.commitResize('split-4');
		ws.resize('split-4', 0, 0.05);
		const answered = split();
		answered['panel-2'].size = 0.7;
		answered['panel-3'].size = 0.3;
		ws.syncFromDoc(answered);
		const root = ws.active.root;
		if (root.kind !== 'split') throw new Error('expected a split');
		expect(root.sizes[0], 'the seam stays where the finger put it').toBeCloseTo(0.75, 6);
	});
});

describe('viewpoint stays here', () => {
	it('routes a sub-patch write to the viewpoint, never to a layout op', async () => {
		const ws = boot();
		ws.setPanelState('panel-2', { subpatchPath: '/inst0' }, 'navigation');
		await Promise.resolve();
		expect(sent(), 'entering a sub-patch is a look, not an edit').toEqual([]);
		expect(ws.viewpoint().paths).toEqual({ 'panel-2': '/inst0' });
		const root = ws.active.root;
		expect(root.kind === 'panel' && root.state).toEqual({ subpatchPath: '/inst0' });
	});

	it('keeps the sub-patch path out of a write that IS shared', async () => {
		const ws = boot();
		ws.setPanelState('panel-2', { subpatchPath: '/inst0' }, 'navigation');
		ws.setPanelSlot('panel-2', 'out');
		await Promise.resolve();
		expect(sent()[0][1], 'a peer must not be dragged into our sub-patch').toEqual({
			page: 'Layout',
			panel: 'panel-2',
			state: { slot: 'out' }
		});
	});

	it('selects a tab and maximizes without sending anything, and each page keeps its own', async () => {
		const two = split();
		two['page-7'] = { kind: 'page', order: 1, name: 'Second' };
		two['panel-8'] = { kind: 'panel', order: 0, parent: 'page-7', size: 1, panel_type: 'console' };
		const ws = boot(two);
		ws.selectTab('page-7');
		ws.toggleMaximize('panel-8');
		await Promise.resolve();
		expect(sent()).toEqual([]);
		expect(ws.state.activeWorkspaceId).toBe('page-7');
		expect(ws.viewpoint().page).toBe('page-7');

		// A maximize belongs to the PAGE it happened on. Looking at another tab used to end it —
		// switching focused that page's first panel, and focusing cleared the one maximize the whole
		// client had — so a user came back to a layout they had already put away.
		ws.selectTab('page-1');
		expect(ws.maximizedPanelId, 'the other page is showing its layout').toBeNull();
		ws.toggleMaximize('panel-3');
		expect(ws.maximizedPanelId).toBe('panel-3');
		ws.selectTab('page-7');
		expect(ws.maximizedPanelId, 'and page 7 is as it was left').toBe('panel-8');
		ws.selectTab('page-1');
		expect(ws.maximizedPanelId, 'so is page 1').toBe('panel-3');

		// Still this client's alone: two pages maximized, and not one byte of it on the wire or in
		// the viewpoint the manager stores and rides into the `.gfi`.
		expect(sent()).toEqual([]);
		expect(Object.keys(ws.viewpoint()).sort()).toEqual(['page', 'panel', 'paths']);
	});

	it('keeps a restored viewpoint through the boundary a fresh session resets across', () => {
		// The boot order, as `_replaceSnapshot` runs it: the snapshot restores the viewpoint, the
		// reset then empties the arrangement so the outgoing session's tree cannot be drawn, and the
		// manager's real document lands only after both. Pruning the viewpoint against that empty
		// middle threw away everything the restore had just put back — and the debounced
		// `set_viewpoint` pushed the loss to the manager.
		const two = split();
		two['page-7'] = { kind: 'page', order: 1, name: 'Second' };
		two['panel-8'] = { kind: 'panel', order: 0, parent: 'page-7', size: 1, panel_type: 'node-editor' };
		const ws = boot({});
		ws.restoreViewpoint({ page: 'page-7', panel: 'panel-8', paths: { 'panel-8': '/inst0' } });
		ws.syncFromDoc({});
		ws.syncFromDoc(two);
		expect(ws.viewpoint(), 'a reload lands where it left off').toEqual({
			page: 'page-7',
			panel: 'panel-8',
			paths: { 'panel-8': '/inst0' }
		});
	});

	it('drops a maximize and a focus a peer’s close took away', () => {
		const ws = boot(split());
		ws.setActive('panel-3');
		ws.toggleMaximize('panel-3');
		ws.syncFromDoc(oneP());
		expect(ws.maximizedPanelId).toBeNull();
		expect(ws.activePanelId).toBe('panel-2');
	});
});

describe('the manager owns the undo step', () => {
	it('records exactly one entry per accepted command, and none for a refusal', async () => {
		const ws = boot(split());
		ws.close('panel-3');
		await Promise.resolve();
		expect(history().length).toBe(1);

		fc.failNext('page_remove_panel');
		ws.close('panel-2');
		await Promise.resolve();
		await Promise.resolve();
		expect(history().length, 'a refused op leaves the two stacks 1:1').toBe(1);
	});
});
