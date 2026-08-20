/**
 * The package, driven standalone.
 *
 * Everything else about this store is tested through goofi's host, which turns each gesture into a
 * manager command — a fine test of goofi, and a poor one of the CONTRACT: a port only proves itself
 * against a second implementation. This is that second one. No socket, no manager, no app: an
 * in-memory host owns the tree, the store raises gestures at it, and what comes back is asserted as
 * an arrangement rather than as a payload.
 *
 * One scenario, in the order a person actually works: open a workspace, split it, give the halves
 * content, resize the seam, tear one off into a tab of its own, come back and close it.
 */
import { describe, expect, it, beforeEach } from 'vitest';
import { memoryHost } from './memoryHost';
import { workspace } from './workspace.svelte';
import { collectPanels, findNode, type LayoutNode, type Workspace } from './model';

/** The arrangement as a shape worth reading in a diff: `panel:type` leaves, `row[…]`/`col[…]`. */
function shape(node: LayoutNode): string {
	if (node.kind === 'panel') return node.panelType;
	const kids = node.children.map(shape).join(' ');
	return `${node.direction === 'row' ? 'row' : 'col'}[${kids}]`;
}
const drawn = (): string[] =>
	workspace().state.workspaces.map((w) => `${w.name}: ${shape(w.root)}`);
const panelsOf = (i = 0): string[] =>
	collectPanels(workspace().state.workspaces[i].root).map((p) => p.id);

/** Settle the store's promise chain: every gesture is raised through an async host. */
const settle = async (): Promise<void> => {
	for (let i = 0; i < 6; i++) await Promise.resolve();
};

function boot(tabs?: Workspace[]) {
	const ws = workspace();
	ws.syncFromDoc([]);
	// Constructing the host is what hands its tree over — it pushes, the way every other host does.
	ws.configureHost(memoryHost({ tabs, defaultPanelType: 'editor', emptyPanelType: 'empty' }));
	return ws;
}

beforeEach(() => {
	workspace().syncFromDoc([]);
});

describe('a workspace with nothing behind it but memory', () => {
	it('draws what the host holds, before a single gesture', async () => {
		const ws = boot();
		await settle();
		expect(drawn()).toEqual(['Tab 1: editor']);
		expect(ws.activePanelId, 'and the focus is somewhere real').toBe(panelsOf()[0]);
	});

	it('carries one person through a working session, and the tree follows', async () => {
		const ws = boot();
		await settle();
		const [first] = panelsOf();

		// Split it, and the fresh half is EMPTY — content is a choice, not an inheritance.
		ws.split(first, 'row');
		await settle();
		expect(drawn()).toEqual(['Tab 1: row[editor empty]']);
		const born = panelsOf()[1];
		expect(ws.activePanelId, 'the focus follows the panel that was just born').toBe(born);

		// Give it content through the same door a panel's own menu uses.
		ws.setType(born, 'console');
		await settle();
		expect(drawn()).toEqual(['Tab 1: row[editor console]']);

		// A seam drag: drawn locally per pointermove, committed once on pointer-up.
		const split = workspace().state.workspaces[0].root;
		expect(split.kind).toBe('split');
		if (split.kind !== 'split') return;
		ws.resize(split.id, 0, 0.2, 800);
		ws.commitResize(split.id);
		await settle();
		const after = workspace().state.workspaces[0].root;
		expect(after.kind === 'split' && after.sizes[0], 'the seam moved and stayed').toBeCloseTo(0.7, 2);

		// Tear the console off onto the tab bar — the landing that names a place with no node.
		ws.dragging = { kind: 'panel', workspaceId: workspace().state.workspaces[0].id, panelId: born };
		ws.dropOn({ newTab: 1 });
		await settle();
		expect(drawn()).toEqual(['Tab 1: editor', 'Tab 2: console']);
		expect(ws.state.activeWorkspaceId, 'and the client follows what it carried').toBe(
			workspace().state.workspaces[1].id
		);
		expect(ws.activePanelId).toBe(born);

		// …and back the other way: dropped onto a panel, it rejoins that tab and takes its own with it.
		ws.dragging = { kind: 'panel', workspaceId: workspace().state.workspaces[1].id, panelId: born };
		ws.dropOn({ panel: first, direction: 'column', placeBefore: false });
		await settle();
		expect(drawn()).toEqual(['Tab 1: col[editor console]']);

		// Closing the last panel of a tab would take the tab; closing one of two just closes it.
		ws.close(born);
		await settle();
		expect(drawn(), 'the survivor is promoted, not left in a split of one').toEqual([
			'Tab 1: editor'
		]);
	});

	it('refuses what it cannot do, and says so rather than half-doing it', async () => {
		const ws = boot();
		await settle();
		const [only] = panelsOf();

		// The last panel on the last tab has nowhere to go.
		ws.close(only);
		await settle();
		expect(drawn(), 'the workspace is never empty').toEqual(['Tab 1: editor']);

		// A panel dropped onto itself is not a move.
		ws.dragging = { kind: 'panel', workspaceId: workspace().state.workspaces[0].id, panelId: only };
		ws.dropOn({ panel: only, direction: 'row', placeBefore: false });
		await settle();
		expect(drawn()).toEqual(['Tab 1: editor']);
		expect(ws.dragging, 'and the drag is spent either way').toBeNull();
	});

	it('adopts a tree it is handed, and keeps minting past its ids', async () => {
		// The other way a consumer starts: with an arrangement it already has. Its ids are the ones
		// every viewpoint is keyed by, so they stand — and the host counts past them rather than
		// minting a second `panel-2` that resolves to the wrong panel.
		boot([
			{ id: 'tab-1', name: 'Saved', root: { kind: 'panel', id: 'panel-2', panelType: 'editor' } }
		]);
		await settle();
		expect(drawn()).toEqual(['Saved: editor']);

		workspace().split('panel-2', 'column');
		await settle();
		const ids = panelsOf();
		expect(ids[0]).toBe('panel-2');
		expect(ids[1], 'the new panel is not a name the tree already used').not.toBe('panel-2');
		expect(findNode(workspace().state.workspaces[0].root, ids[1])).not.toBeNull();
	});
});
