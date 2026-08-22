/**
 * The workspace store as a REPLICA: every frozen gesture leaves as a layout command, nothing edits
 * the arrangement locally, and the viewpoint never leaves as one.
 *
 * These are the tests that pin the cutover. Before it, each gesture rewrote a client-owned tree and
 * `set_layout` pushed the whole blob back debounced; now the tree is the manager's and the gesture
 * is an op, so what a gesture SENDS is the behaviour worth pinning.
 */
import { describe, expect, it, beforeEach } from 'vitest';
import { workspace, type LayoutHost, type LayoutNode, type StackNode } from 'panelty';
import { FakeControl } from '$lib/test/fakeControl';
import { goofiLayoutHost } from './layoutHost';
import { history } from './history.svelte';

const panel = (id: string, panelType: string, state?: unknown): LayoutNode => ({
	kind: 'panel',
	id,
	panelType,
	...(state === undefined ? {} : { state })
});
const strip = (...children: LayoutNode[]): StackNode => ({ kind: 'stack', id: 'stack-1', children });

/** The manager's default arrangement, plus a row split on demand — as the replica reads it off the
 * document, which is the shape the panel system draws. */
const onePage = (): StackNode => strip(panel('panel-2', 'node-editor'));

function split(a = 0.6, b = 0.4, state?: unknown): StackNode {
	return strip({
		kind: 'split',
		id: 'split-4',
		direction: 'row',
		sizes: [a, b],
		children: [panel('panel-2', 'node-editor'), panel('panel-3', 'console', state)]
	});
}

/** A host that answers every gesture with "no" — every member of the port, so a consumer that
 * wires nothing up still gets a workspace it can look at. */
const REFUSING_HOST: LayoutHost = {
	addPanel: async () => null,
	removePanel: async () => false,
	resizeSplit: async () => false,
	setPanel: async () => false,
	movePanel: async () => false
};

let fc: FakeControl;

function boot(root: StackNode | null = onePage()): ReturnType<typeof workspace> {
	fc = new FakeControl();
	const ws = workspace();
	// The store's own seam is the HOST, and goofi's host is what turns a gesture into an op — so
	// these scenarios drive the real one and still assert what reaches the wire.
	ws.configureHost(goofiLayoutHost({ control: () => fc }));
	// The store is a module singleton, so a drag a previous test armed would still be drawing:
	// committing a split that is not the drag's discards it, which is what an abandoned one does.
	ws.commitResize('#none');
	// Through the generation boundary first, the way a real session starts (`_replaceSnapshot`:
	// reset, then the manager's document).
	ws.syncFromDoc(null);
	ws.syncFromDoc(root);
	return ws;
}

/** Drain every pending microtask — a command's reply and whatever its `.then` does with it. */
const settle = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/** The ops sent since boot, as `[op, payload]` pairs. */
function sent(): Array<[string, Record<string, unknown>]> {
	return fc.recordedCalls().map((c) => [c.op, c.payload]);
}

/** The split the fixtures build, for the tests that read its shares back. */
function seam(ws: ReturnType<typeof workspace>): number[] {
	const n = ws.root.children[0];
	if (n.kind !== 'split') throw new Error('expected a split');
	return n.sizes;
}

beforeEach(() => history().reset());

describe('a frozen gesture is a layout command', () => {
	it('rebuilds the tree the manager holds, and holds none of its own', () => {
		const ws = boot(split());
		expect(ws.root.children).toHaveLength(1);
		const root = ws.root.children[0];
		expect(root.kind).toBe('split');
		if (root.kind !== 'split') return;
		expect(root.children.map((c) => c.id)).toEqual(['panel-2', 'panel-3']);
	});

	it('splits through add_panel, carrying the side the drag went', async () => {
		const ws = boot();
		ws.split('panel-2', 'column', true, 0.25);
		await Promise.resolve();
		expect(sent()).toEqual([
			[
				'add_panel',
				{ at: 'panel-2', direction: 'column', place_before: true, ratio: 0.25, index: undefined }
			]
		]);
	});

	it('adds a TAB through the same op, with no direction to split along', async () => {
		const ws = boot();
		ws.add(ws.root.id, { index: 1 });
		await Promise.resolve();
		expect(sent()).toEqual([
			[
				'add_panel',
				{ at: 'stack-1', direction: undefined, place_before: undefined, ratio: undefined, index: 1 }
			]
		]);
	});

	it('closes, retypes and re-binds through the panel ops', async () => {
		const ws = boot(split());
		ws.close('panel-3');
		ws.setType('panel-3', 'viewer');
		ws.linkNodeToPanel('panel-3', 'a1b2');
		await Promise.resolve();
		expect(sent().map(([op]) => op)).toEqual(['remove_panel', 'edit_panel', 'edit_panel']);
		expect(sent()[2][1]).toEqual({ panel: 'panel-3', state: { node: 'a1b2', slot: null } });
	});

	it('settles the slot as it binds, so a picked node is never shown through the old node’s slot', async () => {
		// One write, so it is one undo step — and the SAME write for both doors onto the binding (a
		// node dragged in, a node picked from the bar's dropdown), which is what keeps them honest.
		const ws = boot(split(0.6, 0.4, { node: 'a1b2', slot: 'spectrum' }));
		ws.linkNodeToPanel('panel-3', 'c3d4');
		await Promise.resolve();
		expect(sent()[0][1].state).toEqual({ node: 'c3d4', slot: null });
	});

	it('names only the key a panel write changes, never the bag it read', async () => {
		// A read-modify-write of the whole bag loses whatever a write still in flight put there:
		// `edit_panel` merges, so the client sends the DELTA and the two orders cannot fight.
		const ws = boot(split(0.6, 0.4, { node: 'a1b2', kind: 'line' }));
		ws.setPanelSlot('panel-3', 'out');
		ws.unlinkNodeFromPanel('panel-3');
		await Promise.resolve();
		expect(sent().map(([, p]) => p.state)).toEqual([{ slot: 'out' }, { node: null }]);
	});

	it('writes a panel that lives on a page in the background', async () => {
		// The replica is page-agnostic, and the façade an agent drives addresses any panel. Scoping
		// the lookup to the page in FRONT made a write to any other one silently do nothing.
		const ws = boot(strip(split().children[0], panel('panel-8', 'viewer')));
		ws.linkNodeToPanel('panel-8', 'a1b2');
		await Promise.resolve();
		expect(sent()).toEqual([
			['edit_panel', { panel: 'panel-8', state: { node: 'a1b2', slot: null } }]
		]);
	});

	it('leaves the focus on the panel it dropped, not on the page’s first', async () => {
		// The focus moves when the move is DRAWN, not when the op is answered: until the delta
		// lands the panel is still where it was, and following it there would be following nothing.
		const ws = boot(split());
		ws.setActive('panel-2');
		ws.dragging = { node: 'panel-3' };
		ws.dropOn({ beside: 'panel-2', direction: 'column', placeBefore: false });
		await settle();
		ws.syncFromDoc(
			strip({
				kind: 'split',
				id: 'split-9',
				direction: 'column',
				sizes: [0.5, 0.5],
				children: [panel('panel-2', 'node-editor'), panel('panel-3', 'console')]
			})
		);
		expect(ws.activePanelId, 'the panel the user just moved is the one they are in').toBe('panel-3');
	});

	it('brings a fresh page forward off the id the manager minted, once it arrives', async () => {
		const ws = boot();
		fc.setCallResult('add_panel', 'panel-4');
		ws.add(ws.root.id);
		await settle();
		expect(ws.page, 'not before the page exists to draw').toBe('panel-2');
		ws.syncFromDoc(strip(panel('panel-2', 'node-editor'), panel('panel-4', 'node-editor')));
		expect(ws.page).toBe('panel-4');
		expect(ws.activePanelId).toBe('panel-4');
	});

	it('…and forward all the same when the delta BEATS the answer', async () => {
		// The two race, and an add that also types its panel is two round trips wide — long enough
		// that the page is usually drawn before the id comes back. Waiting for a later sync that
		// nothing is going to send left the new page sitting behind the one it was added from.
		const ws = boot();
		fc.setCallResult('add_panel', 'panel-4');
		ws.add(ws.root.id, { panelType: 'globals' });
		ws.syncFromDoc(strip(panel('panel-2', 'node-editor'), panel('panel-4', 'globals')));
		expect(ws.page, 'not off a delta alone — the id is still in flight').toBe('panel-2');
		await settle();
		expect(ws.page).toBe('panel-4');
		expect(ws.activePanelId).toBe('panel-4');
	});

	it('moves a dragged panel with ONE op, so the drop is one undo step', async () => {
		const ws = boot(split());
		ws.dragging = { node: 'panel-3' };
		ws.dropOn({ beside: 'panel-2', direction: 'column', placeBefore: false });
		await Promise.resolve();
		expect(sent()).toEqual([
			['move_panel', { panel: 'panel-3', to: 'panel-2', direction: 'column', place_before: false }]
		]);
		expect(ws.dragging, 'the drag is spent either way').toBeNull();
	});

	it('drops onto a header as the SAME op, with a place in the strip instead of an axis', async () => {
		// The gesture the tab/panel split used to need a second op for. A drop on a lone panel's
		// header names that panel, and the manager wraps the two in a group.
		const ws = boot(split());
		ws.dragging = { node: 'panel-3' };
		ws.dropOn({ stack: 'panel-2', index: 0 });
		await Promise.resolve();
		expect(sent()).toEqual([['move_panel', { panel: 'panel-3', to: 'panel-2', index: 0 }]]);
	});

	it('tears a panel onto the page strip — the same op again, landing on the ROOT group', async () => {
		const ws = boot(split());
		ws.dragging = { node: 'panel-3' };
		ws.dropOn({ stack: 'stack-1', index: 0 });
		await settle();
		expect(sent()).toEqual([['move_panel', { panel: 'panel-3', to: 'stack-1', index: 0 }]]);
		// A delta that is not this move's own — a peer editing the graph — must not spend the wait:
		// the panel is still on the page it is LEAVING, and settling for that page would leave the
		// torn-off one behind the old one for good.
		ws.syncFromDoc(split());
		expect(ws.activePanelId, 'nothing has moved yet, so nothing is followed').toBe('panel-2');
		ws.syncFromDoc(strip(panel('panel-3', 'console'), panel('panel-2', 'node-editor')));
		expect(ws.page, 'and the page it built comes forward').toBe('panel-3');
		expect(ws.activePanelId).toBe('panel-3');
	});

	it('carries a whole subtree when a split or a group is the thing dragged', async () => {
		const ws = boot(split());
		ws.dragging = { node: 'split-4' };
		ws.dropOn({ stack: 'stack-1', index: 0 });
		await Promise.resolve();
		expect(sent()[0][1], 'a subtree drag names the subtree, not a panel').toMatchObject({
			panel: 'split-4'
		});
	});

	it('closing the page in front moves to its NEIGHBOUR, not to the strip’s first', async () => {
		const ws = boot(
			strip(panel('panel-2', 'node-editor'), panel('panel-4', 'console'), panel('panel-6', 'console'))
		);
		ws.show('stack-1', 'panel-6');
		ws.close('panel-6');
		expect(ws.page, 'the neighbour, before the delta even lands').toBe('panel-4');
		await Promise.resolve();
		expect(sent()).toEqual([['remove_panel', { panel: 'panel-6' }]]);
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
		ws.add(ws.root.id);
		await settle();
		expect(sent(), 'nothing left the client').toEqual([]);
		expect(ws.root.children, 'and the tree it was handed is still drawn').toHaveLength(1);
	});
});

describe('a resize drag draws locally and commits once', () => {
	it('sends nothing while the pointer moves, and one op when it lifts', async () => {
		const ws = boot(split());
		ws.resize('split-4', 0, 0.1);
		ws.resize('split-4', 0, 0.05);
		await Promise.resolve();
		expect(sent(), 'a command per pointermove is exactly what this replaces').toEqual([]);
		expect(seam(ws)[0], 'but the seam moved on screen').toBeCloseTo(0.75, 6);

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
		expect(seam(ws)[0], 'no snap-back between the reply and the delta').toBeCloseTo(0.7, 6);

		ws.syncFromDoc(split(0.7, 0.3));
		expect(seam(ws)).toEqual([0.7, 0.3]);
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
		const ws = boot(split(0.5, 0.5));
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
		ws.syncFromDoc(split(0.7, 0.3));
		expect(seam(ws)[0], 'the seam stays where the finger put it').toBeCloseTo(0.75, 6);
	});
});

describe('viewpoint stays here', () => {
	it('routes a sub-patch write to the viewpoint, never to a layout op', async () => {
		const ws = boot();
		ws.setPanelState('panel-2', { subpatchPath: '/inst0' }, 'navigation');
		await Promise.resolve();
		expect(sent(), 'entering a sub-patch is a look, not an edit').toEqual([]);
		expect(ws.viewpoint().paths).toEqual({ 'panel-2': '/inst0' });
		const shown = ws.root.children[0];
		expect(shown.kind === 'panel' && shown.state).toEqual({ subpatchPath: '/inst0' });
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

	it('shows a page and maximizes without sending anything, and each page keeps its own', async () => {
		const ws = boot(strip(split().children[0], panel('panel-8', 'console')));
		ws.show('stack-1', 'panel-8');
		ws.toggleMaximize('panel-8');
		await Promise.resolve();
		expect(sent()).toEqual([]);
		expect(ws.page).toBe('panel-8');
		expect(ws.viewpoint().showing).toEqual({ 'stack-1': 'panel-8' });

		// A maximize belongs to the PAGE it happened on. Looking at another page used to end it —
		// switching focused that page's first panel, and focusing cleared the one maximize the whole
		// client had — so a user came back to a layout they had already put away.
		ws.show('stack-1', 'split-4');
		expect(ws.maximizedId, 'the other page is showing its layout').toBeNull();
		ws.toggleMaximize('panel-3');
		expect(ws.maximizedId).toBe('panel-3');
		ws.show('stack-1', 'panel-8');
		expect(ws.maximizedId, 'and that page is as it was left').toBe('panel-8');
		ws.show('stack-1', 'split-4');
		expect(ws.maximizedId, 'so is the other').toBe('panel-3');

		// Still this client's alone: two pages maximized, and not one byte of it on the wire or in
		// the viewpoint the manager stores and rides into the `.gfi`.
		expect(sent()).toEqual([]);
		expect(Object.keys(ws.viewpoint()).sort()).toEqual(['panel', 'paths', 'showing']);
	});

	it('maximizes a whole GROUP, which is what a stack makes possible', () => {
		const ws = boot(split());
		ws.toggleMaximize('split-4');
		expect(ws.maximizedId, 'a subtree, not only a leaf').toBe('split-4');
		expect(sent()).toEqual([]);
	});

	it('keeps a restored viewpoint through the boundary a fresh session resets across', () => {
		// The boot order, as `_replaceSnapshot` runs it: the snapshot restores the viewpoint, the
		// reset then empties the arrangement so the outgoing session's tree cannot be drawn, and the
		// manager's real document lands only after both. Pruning the viewpoint against that empty
		// middle threw away everything the restore had just put back — and the debounced
		// `set_viewpoint` pushed the loss to the manager.
		const two = strip(split().children[0], panel('panel-8', 'node-editor'));
		const ws = boot(null);
		ws.restoreViewpoint({
			showing: { 'stack-1': 'panel-8' },
			panel: 'panel-8',
			paths: { 'panel-8': '/inst0' }
		});
		ws.syncFromDoc(null);
		ws.syncFromDoc(two);
		expect(ws.viewpoint(), 'a reload lands where it left off').toEqual({
			showing: { 'stack-1': 'panel-8' },
			panel: 'panel-8',
			paths: { 'panel-8': '/inst0' }
		});
	});

	it('drops a maximize and a focus a peer’s close took away', () => {
		const ws = boot(split());
		ws.setActive('panel-3');
		ws.toggleMaximize('panel-3');
		ws.syncFromDoc(onePage());
		expect(ws.maximizedId).toBeNull();
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
