import { describe, it, expect, beforeEach } from 'vitest';
import { workspace } from './workspace.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { captureNavContext, restoreNavContext } from './navContext';

describe('NavContext capture/restore', () => {
	beforeEach(() => {
		workspace().reset();
		selection().forgetAll();
	});

	it('captures the active workspace, panel, and per-panel selection', () => {
		const ws = workspace();
		const panelId = ws.activePanelId!;
		selection().selectNodes(panelId, ['osc0', 'buffer0']);
		const ctx = captureNavContext();
		expect(ctx.activeWorkspaceId).toBe(ws.state.activeWorkspaceId);
		expect(ctx.activePanelId).toBe(panelId);
		expect(ctx.selection[panelId].nodes.sort()).toEqual(['buffer0', 'osc0']);
	});

	it('restore re-applies the selection and active panel', async () => {
		const ws = workspace();
		const panelId = ws.activePanelId!;
		selection().selectNodes(panelId, ['osc0']);
		const ctx = captureNavContext();
		selection().clear(panelId);
		expect(selection().nodes(panelId).size).toBe(0);
		await restoreNavContext(ctx);
		expect([...selection().nodes(panelId)]).toEqual(['osc0']);
		expect(ws.activePanelId).toBe(panelId);
	});

	it('falls back to a live editor panel when the recorded panel no longer exists', async () => {
		const ws = workspace();
		const oldPanel = ws.activePanelId!;
		selection().selectNodes(oldPanel, ['osc0']);
		const ctx = captureNavContext();
		// The change being undone closed the recorded panel: a fresh workspace
		// replaces it with a different panel id.
		ws.reset();
		const newPanel = ws.activePanelId!;
		expect(newPanel).not.toBe(oldPanel);
		await restoreNavContext(ctx);
		// Not a no-op: a live editor is focused and the recorded selection lands there.
		expect(ws.activePanelId).toBe(newPanel);
		expect([...selection().nodes(newPanel)]).toEqual(['osc0']);
	});

	it('captures and restores a panel sub-patch path (enteredPath)', async () => {
		const ws = workspace();
		const panelId = ws.activePanelId!;
		ws.setPanelState(panelId, { subpatchPath: '/subpatch0' });
		const ctx = captureNavContext();
		expect(ctx.enteredPath[panelId]).toEqual(['subpatch0']);
		ws.setPanelState(panelId, { subpatchPath: '/' }); // navigate out
		await restoreNavContext(ctx);
		const root = ws.active.root;
		const state = root.kind === 'panel' ? (root.state as { subpatchPath?: string }) : {};
		expect(state.subpatchPath).toBe('/subpatch0');
	});
});
