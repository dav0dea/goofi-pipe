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
import { goofiLayoutHost } from '$lib/stores/layoutHost';
import { history } from '$lib/stores/history.svelte';
import type { Workspace } from './model';
import type { LayoutHost } from './host';

/** The manager's default arrangement, plus a row split on demand — as the replica reads it off the
 * document, which is the shape the panel system draws. */
function oneTab(): Workspace[] {
	return [
		{
			id: 'tab-1',
			name: 'Tab 1',
			root: { kind: 'panel', id: 'panel-2', panelType: 'node-editor' }
		}
	];
}
function split(): Workspace[] {
	return [
		{
			id: 'tab-1',
			name: 'Tab 1',
			root: {
				kind: 'split',
				id: 'split-4',
				direction: 'row',
				sizes: [0.6, 0.4],
				children: [
					{ kind: 'panel', id: 'panel-2', panelType: 'node-editor' },
					{ kind: 'panel', id: 'panel-3', panelType: 'console' }
				]
			}
		}
	];
}

/** One tab holding one panel — the shape every extra tab in these scenarios takes. */
function tab(id: string, name: string, panelId: string, panelType: string): Workspace {
	return { id, name, root: { kind: 'panel', id: panelId, panelType } };
}

/** `split()` with `panel-3`'s bag already set, which is what a rebind is asserted against. */
function bound(state: unknown): Workspace[] {
	const tabs = split();
	const root = tabs[0].root;
	if (root.kind === 'split') root.children[1] = { ...root.children[1], state } as typeof root.children[1];
	return tabs;
}

/** `split()` with the seam moved, as the manager's answer to a resize lands it. */
function resized(a: number, b: number): Workspace[] {
	const tabs = split();
	const root = tabs[0].root;
	if (root.kind === 'split') root.sizes = [a, b];
	return tabs;
}

/** A host that answers every gesture with "no". `tabFromPanel` is present so a spread can remove
 * it, which is what the composed-only contract looks like from a consumer's side. */
const REFUSING_HOST: LayoutHost = {
	addTab: async () => null,
	removeTab: async () => false,
	renameTab: async () => false,
	reorderTab: async () => false,
	splitPanel: async () => null,
	removePanel: async () => false,
	resizeSplit: async () => false,
	setPanel: async () => false,
	movePanel: async () => false,
	tabFromPanel: async () => null
};

let fc: FakeControl;

function boot(tabs: Workspace[] = oneTab()): ReturnType<typeof workspace> {
	fc = new FakeControl();
	const ws = workspace();
	// The store's own seam is the HOST, and goofi's host is what turns a gesture into an op — so
	// these scenarios drive the real one and still assert what reaches the wire.
	ws.configureHost(goofiLayoutHost({ control: () => fc, tabs: () => ws.state.workspaces }));
	// The store is a module singleton, so a drag a previous test armed would still be drawing:
	// committing a split that is not the drag's discards it, which is what an abandoned one does.
	ws.commitResize('#none');
	// Through the generation boundary first, the way a real session starts (`_replaceSnapshot`:
	// reset, then the manager's document). Without it the store carries the previous test's name
	// claims into this one, and the next claimed tab name depends on which tests ran before.
	ws.syncFromDoc([]);
	ws.syncFromDoc(tabs);
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

	it('splits through split_panel, carrying the side the drag went', async () => {
		const ws = boot();
		ws.split('panel-2', 'column', true, 0.25);
		await Promise.resolve();
		expect(sent()).toEqual([
			[
				'split_panel',
				{ panel: 'panel-2', direction: 'column', place_before: true, ratio: 0.25 }
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
			'remove_panel',
			'set_panel',
			'set_panel'
		]);
		expect(sent()[2][1]).toEqual({
			panel: 'panel-3',
			state: { node: 'a1b2', slot: null }
		});
	});

	it('settles the slot as it binds, so a picked node is never shown through the old node’s slot', async () => {
		// One write, so it is one undo step — and the SAME write for both doors onto the binding (a
		// node dragged in, a node picked from the bar's dropdown), which is what keeps them honest.
		const ws = boot(bound({ node: 'a1b2', slot: 'spectrum' }));
		ws.linkNodeToPanel('panel-3', 'c3d4');
		await Promise.resolve();
		expect(sent()[0][1].state).toEqual({ node: 'c3d4', slot: null });
	});

	it('names only the key a panel write changes, never the bag it read', async () => {
		// A read-modify-write of the whole bag loses whatever a write still in flight put there:
		// `set_panel` merges, so the client sends the DELTA and the two orders cannot fight.
		const ws = boot(bound({ node: 'a1b2', kind: 'line' }));
		ws.setPanelSlot('panel-3', 'out');
		ws.unlinkNodeFromPanel('panel-3');
		await Promise.resolve();
		expect(sent().map(([, p]) => p.state)).toEqual([{ slot: 'out' }, { node: null }]);
	});

	it('writes a panel that lives on a page in the background', async () => {
		// The replica is page-agnostic, and the façade an agent drives addresses any panel. Scoping
		// the lookup to the page in FRONT made a write to any other one silently do nothing.
		const ws = boot([...split(), tab('tab-7', 'Second', 'panel-8', 'viewer')]);
		ws.linkNodeToPanel('panel-8', 'a1b2');
		await Promise.resolve();
		expect(sent()).toEqual([
			['set_panel', { panel: 'panel-8', state: { node: 'a1b2', slot: null } }]
		]);
	});

	it('leaves the focus on the panel it dropped, not on the page’s first', async () => {
		const ws = boot(split());
		ws.setActive('panel-2');
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropOnPanel('panel-2', 'column', false);
		await settle();
		expect(ws.activePanelId, 'the panel the user just moved is the one they are in').toBe('panel-3');
	});

	it('brings a fresh tab forward off the ids the manager minted, once they arrive', async () => {
		const ws = boot();
		fc.setCallResult('add_tab', { tab: 'tab-3', panel: 'panel-4' });
		ws.addTab();
		await settle();
		expect(ws.state.activeWorkspaceId, 'not before the tab exists to draw').toBe('tab-1');
		ws.syncFromDoc([...oneTab(), tab('tab-3', 'Tab 2', 'panel-4', 'node-editor')]);
		expect(ws.state.activeWorkspaceId).toBe('tab-3');
		expect(ws.activePanelId).toBe('panel-4');
	});

	it('moves a dragged panel with ONE op, so the drop is one undo step', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropOnPanel('panel-2', 'column', false);
		await Promise.resolve();
		expect(sent()).toEqual([
			[
				'insert_at_panel',
				{
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
		ws.dragging = { kind: 'tab', workspaceId: 'tab-1' };
		ws.dropOnPanel('panel-2', 'row', false);
		await Promise.resolve();
		expect(sent()[0][1], 'a tab drag names the page’s root, not a panel').toMatchObject({
			subtree: 'split-4'
		});
	});

	it('tears a panel onto the tab bar as one page built around it', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropPanelOnTabBar(0);
		await Promise.resolve();
		expect(sent()).toEqual([
			['add_tab', { name: 'Tab 2', index: 0, subtree: 'panel-3' }]
		]);
	});

	it('closing the tab in front moves to its NEIGHBOUR, not to the strip’s first', async () => {
		const ws = boot([
			...oneTab(),
			tab('tab-3', 'Two', 'panel-4', 'console'),
			tab('tab-5', 'Three', 'panel-6', 'console')
		]);
		ws.selectTab('tab-5');
		ws.closeTab('tab-5');
		expect(ws.state.activeWorkspaceId, 'the neighbour, before the delta even lands').toBe('tab-3');
		await Promise.resolve();
		expect(sent()).toEqual([['remove_tab', { tab: 'tab-5' }]]);
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
		expect(names, 'and none of them is the name the tab already has').not.toContain('Tab 1');
	});

	it('draws and refuses when nothing has installed a host', async () => {
		// The panel system holds no tree and writes nothing itself, so a consumer that has not wired
		// one up should get a workspace it can look at rather than an exception per click. Every
		// gesture below is a no-op; none of them throws, and none reaches the wire.
		fc = new FakeControl();
		const ws = workspace();
		ws.configureHost(REFUSING_HOST);
		ws.syncFromDoc(split());
		ws.split('panel-2', 'row');
		ws.close('panel-3');
		ws.setType('panel-2', 'console');
		ws.addTab();
		ws.renameTab('tab-1', 'Signals');
		await settle();
		expect(sent(), 'nothing left the client').toEqual([]);
		expect(ws.state.workspaces, 'and the tree it was handed is still drawn').toHaveLength(1);
	});

	it('offers the tear-off only where the host can express it', async () => {
		// `tabFromPanel` is the one gesture that spans tabs AND panels, so it is optional on the
		// port: a host without it does not fail the drag, it never offers one. The drag is spent
		// either way — a gesture that goes nowhere must not leave the pointer armed.
		fc = new FakeControl();
		const ws = workspace();
		ws.configureHost({ ...REFUSING_HOST, tabFromPanel: undefined });
		ws.syncFromDoc(split());
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropPanelOnTabBar(0);
		await settle();
		expect(sent()).toEqual([]);
		expect(ws.dragging, 'the drag is spent either way').toBeNull();
		// …and the strip is told, so it never draws a drop it would have to refuse.
		expect(ws.canTearOff).toBe(false);

		ws.configureHost(REFUSING_HOST);
		expect(ws.canTearOff, 'a composed host offers it').toBe(true);
	});

	it('addresses a tab by its ID, never by the label it happens to wear', async () => {
		// A label is what a tab HOLDS. Addressing by it meant every op re-derived a name from the id
		// it already had, and a rename landing between the two made the next op miss.
		const ws = boot(split());
		ws.renameTab('tab-1', 'Signals');
		ws.closeTab('tab-1');
		ws.reorderTab(0, 0);
		await Promise.resolve();
		expect(sent()).toEqual([
			['rename_tab', { tab: 'tab-1', name: 'Signals' }],
			['remove_tab', { tab: 'tab-1' }],
			['reorder_tab', { tab: 'tab-1', to_index: 0 }]
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
		expect(op).toBe('resize_split');
		expect(payload).toMatchObject({ split: 'split-4' });
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

		const answered = resized(0.7, 0.3);
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
		const even = resized(0.5, 0.5);
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
		const answered = resized(0.7, 0.3);
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
			panel: 'panel-2',
			state: { slot: 'out' }
		});
	});

	it('selects a tab and maximizes without sending anything, and each page keeps its own', async () => {
		const two = [...split(), tab('tab-7', 'Second', 'panel-8', 'console')];
		const ws = boot(two);
		ws.selectTab('tab-7');
		ws.toggleMaximize('panel-8');
		await Promise.resolve();
		expect(sent()).toEqual([]);
		expect(ws.state.activeWorkspaceId).toBe('tab-7');
		expect(ws.viewpoint().tab).toBe('tab-7');

		// A maximize belongs to the PAGE it happened on. Looking at another tab used to end it —
		// switching focused that page's first panel, and focusing cleared the one maximize the whole
		// client had — so a user came back to a layout they had already put away.
		ws.selectTab('tab-1');
		expect(ws.maximizedPanelId, 'the other page is showing its layout').toBeNull();
		ws.toggleMaximize('panel-3');
		expect(ws.maximizedPanelId).toBe('panel-3');
		ws.selectTab('tab-7');
		expect(ws.maximizedPanelId, 'and page 7 is as it was left').toBe('panel-8');
		ws.selectTab('tab-1');
		expect(ws.maximizedPanelId, 'so is page 1').toBe('panel-3');

		// Still this client's alone: two pages maximized, and not one byte of it on the wire or in
		// the viewpoint the manager stores and rides into the `.gfi`.
		expect(sent()).toEqual([]);
		expect(Object.keys(ws.viewpoint()).sort()).toEqual(['panel', 'paths', 'tab']);
	});

	it('keeps a restored viewpoint through the boundary a fresh session resets across', () => {
		// The boot order, as `_replaceSnapshot` runs it: the snapshot restores the viewpoint, the
		// reset then empties the arrangement so the outgoing session's tree cannot be drawn, and the
		// manager's real document lands only after both. Pruning the viewpoint against that empty
		// middle threw away everything the restore had just put back — and the debounced
		// `set_viewpoint` pushed the loss to the manager.
		const two = [...split(), tab('tab-7', 'Second', 'panel-8', 'node-editor')];
		const ws = boot([]);
		ws.restoreViewpoint({ tab: 'tab-7', panel: 'panel-8', paths: { 'panel-8': '/inst0' } });
		ws.syncFromDoc([]);
		ws.syncFromDoc(two);
		expect(ws.viewpoint(), 'a reload lands where it left off').toEqual({
			tab: 'tab-7',
			panel: 'panel-8',
			paths: { 'panel-8': '/inst0' }
		});
	});

	it('a name claimed for a tab the load took away is free again', async () => {
		// A claim reserves a tab name until the replica shows it, so six taps on ＋ do not ask for the
		// same free name six times. It is the one thing the generation boundary genuinely ends: the
		// name was reserved against the OUTGOING strip, and a patch loaded out from under it means
		// the page it was claimed for is never coming. Left standing, `_claimName` skipped that name
		// for the rest of the session.
		const ws = boot();
		ws.addTab();
		await settle();
		expect(sent()).toEqual([['add_tab', { name: 'Tab 2' }]]);

		ws.syncFromDoc([]);
		ws.syncFromDoc(oneTab());
		ws.addTab();
		await settle();
		expect(sent()[1], 'the new session offers the name again').toEqual([
			'add_tab',
			{ name: 'Tab 2' }
		]);
	});

	it('drops a maximize and a focus a peer’s close took away', () => {
		const ws = boot(split());
		ws.setActive('panel-3');
		ws.toggleMaximize('panel-3');
		ws.syncFromDoc(oneTab());
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

		fc.failNext('remove_panel');
		ws.close('panel-2');
		await Promise.resolve();
		await Promise.resolve();
		expect(history().length, 'a refused op leaves the two stacks 1:1').toBe(1);
	});
});
