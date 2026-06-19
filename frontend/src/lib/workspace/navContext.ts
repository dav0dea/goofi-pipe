/**
 * Navigation/focus context for undo/redo. Each recorded action snapshots WHERE
 * it happened; on undo/redo we restore that context first so the undone/redone
 * change is highlighted in the right tab/panel/sub-patch.
 *
 * Phase 2 captures the tab + active panel. Phase 5 enriches this with per-panel
 * sub-patch `enteredPath`, selection restore, and the highlight pulse.
 */
import { workspace } from './workspace.svelte';
import type { NavContext } from '$lib/stores/history.svelte';

export function captureNavContext(): NavContext {
	const ws = workspace();
	return {
		activeWorkspaceId: ws.state.activeWorkspaceId,
		activePanelId: ws.activePanelId,
		enteredPath: {},
		selection: {}
	};
}

export async function restoreNavContext(ctx: NavContext): Promise<void> {
	const ws = workspace();
	if (ctx.activeWorkspaceId && ctx.activeWorkspaceId !== ws.state.activeWorkspaceId) {
		ws.selectTab(ctx.activeWorkspaceId);
	}
	if (ctx.activePanelId) ws.setActive(ctx.activePanelId);
	// Phase 5: restore enteredPath + selection + flash the changed node/param.
}
