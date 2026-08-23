/**
 * The workspace store as a REPLICA: every frozen gesture leaves as a layout command, nothing edits
 * the arrangement locally, and the viewpoint never leaves as one.
 *
 * These are the tests that pin the cutover. Before it, each gesture rewrote a client-owned tree and
 * `set_layout` pushed the whole blob back debounced; now the tree is the manager's and the gesture
 * is an op, so what a gesture SENDS is the behaviour worth pinning.
 */
import { describe, expect, it, beforeEach } from 'vitest';
import { workspace, type LayoutHost, type Workspace } from 'panelty';
import { FakeControl } from '$lib/test/fakeControl';
import { goofiLayoutHost } from './layoutHost';
import { history } from './history.svelte';

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

/** A host that answers every gesture with "no" — every member of the port, so a consumer that
 * wires nothing up still gets a workspace it can look at. */
const REFUSING_HOST: LayoutHost = {
	addTab: async () => null,
	removeTab: async () => false,
	renameTab: async () => false,
	reorderTab: async () => false,
	splitPanel: async () => null,
	removePanel: async () => false,
	resizeSplit: async () => false,
	setPanel: async () => false,
	movePanel: async () => false
};

let fc: FakeControl;

function boot(tabs: Workspace[] = oneTab()): ReturnType<typeof workspace> {
	fc = new FakeControl();
	const ws = workspace();
	// The store's own seam is the HOST, and goofi's host is what turns a gesture into an op — so
	// these scenarios drive the real one and still assert what reaches the wire.
	ws.configureHost(goofiLayoutHost({ control: () => fc }));
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

	it('splits through place_panel, carrying the SIDE the drag went', async () => {
		// The panel system raises an axis and a half; the op takes one word for the pair, so the two
		// cannot disagree on the way over.
		const ws = boot();
		ws.split('panel-2', 'column', true, 0.25);
		await Promise.resolve();
		expect(sent()).toEqual([['place_panel', { to: 'panel-2', direction: 'top', ratio: 0.25 }]]);
	});

	it('…and every other side maps to its own word', async () => {
		const ws = boot();
		ws.split('panel-2', 'column', false);
		ws.split('panel-2', 'row', true);
		ws.split('panel-2', 'row', false);
		await Promise.resolve();
		expect(sent().map(([, p]) => p.direction)).toEqual(['bottom', 'left', 'right']);
	});

	it('closes, retypes and re-binds through the page ops', async () => {
		const ws = boot(split());
		ws.close('panel-3');
		ws.setType('panel-3', 'viewer');
		ws.linkNodeToPanel('panel-3', 'a1b2');
		await Promise.resolve();
		expect(sent().map(([op]) => op)).toEqual(['remove_panel', 'edit_panel', 'edit_panel']);
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
		// `edit_panel` merges, so the client sends the DELTA and the two orders cannot fight.
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
			['edit_panel', { panel: 'panel-8', state: { node: 'a1b2', slot: null } }]
		]);
	});

	it('leaves the focus on the panel it dropped, not on the page’s first', async () => {
		const ws = boot(split());
		ws.setActive('panel-2');
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropOn({ panel: 'panel-2', direction: 'column', placeBefore: false });
		await settle();
		expect(ws.activePanelId, 'the panel the user just moved is the one they are in').toBe('panel-3');
	});

	it('brings a fresh tab forward off the ids the manager minted, once they arrive', async () => {
		const ws = boot();
		fc.setCallResult('place_panel', { tab: 'tab-3', id: 'panel-4' });
		ws.addTab();
		await settle();
		expect(ws.state.activeWorkspaceId, 'not before the tab exists to draw').toBe('tab-1');
		ws.syncFromDoc([...oneTab(), tab('tab-3', 'Tab 2', 'panel-4', 'node-editor')]);
		expect(ws.state.activeWorkspaceId).toBe('tab-3');
		expect(ws.activePanelId).toBe('panel-4');
	});

	it('…and forward all the same when the delta BEATS the answer', async () => {
		// The two race, and a `place_panel` that also types its panel is two round trips wide — long
		// enough that the tab is usually drawn before the ids come back. Waiting for a later sync that
		// nothing is going to send left the new tab sitting behind the one it was added from.
		const ws = boot();
		fc.setCallResult('place_panel', { tab: 'tab-3', id: 'panel-4' });
		ws.addTab('globals');
		ws.syncFromDoc([...oneTab(), tab('tab-3', 'Tab 2', 'panel-4', 'globals')]);
		expect(ws.state.activeWorkspaceId, 'not off a delta alone — the ids are still in flight').toBe(
			'tab-1'
		);
		await settle();
		expect(ws.state.activeWorkspaceId).toBe('tab-3');
		expect(ws.activePanelId).toBe('panel-4');
	});

	it('moves a dragged panel with ONE op, so the drop is one undo step', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropOn({ panel: 'panel-2', direction: 'column', placeBefore: false });
		await Promise.resolve();
		expect(sent()).toEqual([
			['place_panel', { panel: 'panel-3', to: 'panel-2', direction: 'bottom' }]
		]);
		expect(ws.dragging, 'the drag is spent either way').toBeNull();
	});

	it('carries a whole tab’s subtree when the tab itself is dragged', async () => {
		const ws = boot(split());
		ws.dragging = { kind: 'tab', workspaceId: 'tab-1' };
		ws.dropOn({ panel: 'panel-2', direction: 'row', placeBefore: false });
		await Promise.resolve();
		expect(sent()[0][1], 'a tab drag names the page’s root, not a panel').toMatchObject({
			panel: 'split-4'
		});
	});

	it('tears a panel onto the tab bar as one page built around it', async () => {
		// The same MOVE as a drop onto a panel — the landing is a place that does not exist yet, which
		// is the only thing that makes it a second op rather than a second method.
		const ws = boot(split());
		ws.dragging = { kind: 'panel', workspaceId: 'tab-1', panelId: 'panel-3' };
		ws.dropOn({ newTab: 0 });
		await settle();
		expect(sent()).toEqual([['place_panel', { panel: 'panel-3', index: 0 }]]);
		// A delta that is not this move's own — a peer editing the graph — must not spend the wait:
		// the panel is still drawn on the tab it is LEAVING, and settling for that tab would leave the
		// torn-off one behind the old one for good.
		ws.syncFromDoc(split());
		expect(ws.activePanelId, 'nothing has moved yet, so nothing is followed').toBe('panel-2');
		ws.syncFromDoc([tab('tab-9', 'Tab 2', 'panel-3', 'console'), ...oneTab()]);
		expect(ws.state.activeWorkspaceId, 'and the tab it built comes forward').toBe('tab-9');
		expect(ws.activePanelId).toBe('panel-3');
	});

	it('refuses to build a tab around a TAB dropped on the bar — that is a reorder', async () => {
		// The strip commits a tab drop itself. Taking it as a tear-off would rebuild the tab it already
		// is, under a fresh id and a fresh name, and every viewpoint keyed by the old id would drop.
		const ws = boot(split());
		ws.dragging = { kind: 'tab', workspaceId: 'tab-1' };
		ws.dropOn({ newTab: 0 });
		await settle();
		expect(sent()).toEqual([]);
		expect(ws.dragging, 'the drag is spent either way').toBeNull();
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
		expect(sent()).toEqual([['remove_panel', { panel: 'tab-5' }]]);
	});

	it('asks for NO tab name — the strip it would have to guess at is the manager’s', async () => {
		// This used to claim `Tab n` here, with an in-flight reservation set, because three taps
		// inside one round trip would otherwise all ask for the same free name. The manager sees the
		// whole strip under the lock the add runs on, so the race has nowhere to happen.
		const ws = boot();
		ws.addTab();
		ws.addTab();
		ws.addTab();
		await Promise.resolve();
		await Promise.resolve();
		expect(sent().map(([op]) => op), 'three taps, three requests').toEqual([
			'place_panel',
			'place_panel',
			'place_panel'
		]);
		expect(
			sent().some(([, p]) => 'name' in p),
			'and not one of them names a tab'
		).toBe(false);
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

	it('addresses a tab by its ID, never by the label it happens to wear', async () => {
		// A label is what a tab HOLDS. Addressing by it meant every op re-derived a name from the id
		// it already had, and a rename landing between the two made the next op miss.
		const ws = boot(split());
		ws.renameTab('tab-1', 'Signals');
		ws.closeTab('tab-1');
		ws.reorderTab(0, 0);
		await Promise.resolve();
		expect(sent()).toEqual([
			['edit_panel', { panel: 'tab-1', name: 'Signals' }],
			['remove_panel', { panel: 'tab-1' }],
			['place_panel', { panel: 'tab-1', index: 0 }]
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
		expect(op).toBe('edit_panel');
		expect(payload).toMatchObject({ panel: 'split-4' });
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
