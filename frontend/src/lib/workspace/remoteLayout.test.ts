import { describe, it, expect, beforeEach } from 'vitest';
import { workspace } from './workspace.svelte';
import { history } from '$lib/stores/history.svelte';
import { collectPanels } from './model';
import type { WorkspaceState } from './model';
import { asStateObject } from './panelState';

/**
 * Applying a PEER's authored arrangement.
 *
 * The layout is one opaque blob, so a peer's push carries both its panel TREE and wherever that
 * peer happened to be looking when it authored. Only the tree is shared: `hydrate` replaces the
 * state wholesale, which is right for a patch that just loaded and wrong for a desktop adding a
 * panel while a phone sits three sub-patches deep. So this merges — remote structure, local
 * viewpoint — and latches, so the shell does not push the peer's own arrangement straight back.
 */

/** A snapshot both "clients" boot from: on `hello` each hydrates the SAME stored layout, so their
 * panel ids agree — which is what lets a merge carry a local viewpoint onto a surviving panel. */
function shared(): WorkspaceState {
	workspace().reset();
	return workspace().serialize();
}

/** The blob a peer would push after doing `author` on top of `base`, leaving the local store back
 * on `base` — the two-client setup, driven through the one singleton store. */
function peerAuthored(base: WorkspaceState, author: (ws: ReturnType<typeof workspace>) => void): WorkspaceState {
	const ws = workspace();
	ws.hydrate(structuredClone(base));
	author(ws);
	const remote = ws.serialize();
	ws.hydrate(structuredClone(base));
	ws.takeLayoutIntent();
	ws.takeRemoteApplied();
	return remote;
}

function panelIds(): string[] {
	return workspace().state.workspaces.flatMap((w) => collectPanels(w.root).map((p) => p.id));
}

describe('a peer’s authored layout', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
		workspace().takeLayoutIntent();
		workspace().takeRemoteApplied();
	});

	it('brings the peer’s structure over', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		workspace().applyRemoteLayout(remote);
		expect(panelIds(), 'the split the peer made is now ours too').toHaveLength(2);
	});

	it('keeps this client’s own sub-patch position on a panel that survived', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		const ws = workspace();
		const mine = ws.activePanelId!;
		ws.setPanelState(mine, { subpatchPath: '/inst0' }, 'navigation');

		ws.applyRemoteLayout(remote);

		const panel = collectPanels(ws.active.root).find((p) => p.id === mine);
		expect(asStateObject(panel?.state).subpatchPath, 'a peer adding a panel must not climb us out').toBe(
			'/inst0'
		);
	});

	it('does NOT import the peer’s sub-patch position onto a panel we are already looking at', () => {
		const base = shared();
		const mine = workspace().activePanelId!;
		const remote = peerAuthored(base, (ws) => {
			ws.setPanelState(mine, { subpatchPath: '/inst0' }, 'navigation');
			ws.split(mine, 'row');
		});

		const ws = workspace();
		ws.applyRemoteLayout(remote);

		const panel = collectPanels(ws.active.root).find((p) => p.id === mine);
		expect(asStateObject(panel?.state).subpatchPath, 'where the peer was looking is not ours').toBeUndefined();
	});

	it('keeps this client’s active layout tab', () => {
		const base = shared();
		const ws = workspace();
		const mine = ws.state.activeWorkspaceId;
		// The peer adds a tab, which also brings ITS tab to the front. The tab travels; the front
		// one does not.
		const remote = peerAuthored(base, (w) => w.addTab());

		ws.applyRemoteLayout(remote);

		expect(ws.state.workspaces, 'the peer’s new tab arrived').toHaveLength(2);
		expect(ws.state.activeWorkspaceId, 'we are still looking at our own tab').toBe(mine);
	});

	it('falls back to the peer’s active tab when ours is the one it closed', () => {
		const base = shared();
		const ws = workspace();
		ws.addTab(); // two tabs, ours is the second
		const both = ws.serialize();
		const doomed = ws.state.activeWorkspaceId;
		const remote = peerAuthored(both, (w) => w.closeTab(doomed));
		ws.takeLayoutIntent();

		ws.applyRemoteLayout(remote);

		expect(ws.state.workspaces.some((w) => w.id === doomed), 'the tab is gone').toBe(false);
		expect(ws.state.activeWorkspaceId, 'we land on the tab that is left').toBe(remote.activeWorkspaceId);
	});

	it('refuses a malformed blob rather than blanking the workspace', () => {
		const before = workspace().serialize();
		workspace().applyRemoteLayout({ workspaces: [] });
		expect(workspace().serialize(), 'a blob we cannot read is not an arrangement').toEqual(before);
	});

	it('is nobody’s edit — it must not dirty the patch', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		const ws = workspace();
		ws.applyRemoteLayout(remote);
		expect(ws.takeLayoutIntent(), 'the peer’s own push already dirtied it, once').toBe('navigation');
	});

	it('reseeds ids so a panel we mint next cannot collide with the peer’s', () => {
		const ws = workspace();
		ws.applyRemoteLayout({
			workspaces: [
				{ id: 'ws-40', name: 'Layout', root: { kind: 'panel', id: 'panel-42', panelType: 'node-editor' } }
			],
			activeWorkspaceId: 'ws-40'
		});
		ws.split(ws.activePanelId!, 'row');
		const minted = panelIds().filter((id) => id !== 'panel-42');
		expect(minted, 'exactly one new panel').toHaveLength(1);
		expect(Number(/-(\d+)$/.exec(minted[0])![1]), 'minted past every id the peer already used').toBeGreaterThan(42);
	});
});

describe('the echo the peer’s layout must not cause', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
		workspace().takeLayoutIntent();
		workspace().takeRemoteApplied();
	});

	it('latches, so the shell drops the push a remote apply scheduled', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		const ws = workspace();
		ws.applyRemoteLayout(remote);
		expect(ws.takeRemoteApplied(), 'the manager already holds this arrangement').toBe(true);
		expect(ws.takeRemoteApplied(), 'and the latch is spent').toBe(false);
	});

	it('does not latch a LOCAL write that lands in the same debounce window', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		const ws = workspace();
		ws.applyRemoteLayout(remote);
		ws.split(ws.activePanelId!, 'column'); // our own edit, inside the 400ms window
		expect(ws.takeRemoteApplied(), 'our edit still has to reach the manager').toBe(false);
	});

	it('does not latch a local NAVIGATION write either — it still has to persist', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		const ws = workspace();
		ws.applyRemoteLayout(remote);
		ws.setPanelState(ws.activePanelId!, { subpatchPath: '/inst0' }, 'navigation');
		expect(ws.takeRemoteApplied()).toBe(false);
	});

	it('does not latch a wholesale load that lands after it', () => {
		const base = shared();
		const remote = peerAuthored(base, (ws) => ws.split(ws.activePanelId!, 'row'));
		const ws = workspace();
		ws.applyRemoteLayout(remote);
		ws.hydrate(structuredClone(base)); // a patch loaded — the manager needs to hear about it
		expect(ws.takeRemoteApplied()).toBe(false);
	});
});

/**
 * Hydrating a blob whose WORKSPACE id outruns its node ids — exactly what `dropPanelOnTabBar`
 * saves, since it mints a fresh `uid('ws')` around a panel node that already existed. After a
 * reload the module counter starts over, so the reseed is the only thing standing between the
 * next `addTab` and a duplicate tab id — and a duplicate is worse than inert: `closeTab` and
 * `renameTab` filter/map by id, so one ✕ closes both tabs.
 */
describe('adopting a saved layout whose tab id outruns its panel ids', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
	});

	it('mints a distinct id for the next tab', () => {
		const ws = workspace();
		ws.hydrate({
			workspaces: [
				{ id: 'ws-9000', name: 'Layout', root: { kind: 'panel', id: 'panel-8999', panelType: 'node-editor' } }
			],
			activeWorkspaceId: 'ws-9000'
		});
		ws.addTab();
		const ids = ws.state.workspaces.map((w) => w.id);
		expect(new Set(ids).size, `two tabs, two ids (got ${ids.join(', ')})`).toBe(ids.length);
	});
});
