/**
 * Navigation/focus context for undo/redo. Each recorded action snapshots WHERE
 * it happened — active tab + panel, each editor's sub-patch depth, and the
 * selection — so on undo/redo we reorient there and highlight the change.
 *
 * Navigation itself is never tracked; this only *restores* it. Restoring the
 * sub-patch depth works through the same `panel.state.subpatchPath` seam the
 * NodeEditorPanel reads reactively, so writing it makes the editor follow.
 */
import { workspace } from './workspace.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { collectPanels, findPanel } from './model';
import { asStateObject } from './panelState';
import type { NavContext } from '$lib/stores/history.svelte';

function pathToArray(p: unknown): string[] {
	return typeof p === 'string' ? p.split('/').filter(Boolean) : [];
}
function arrayToPath(a: string[]): string {
	return '/' + a.join('/');
}

export function captureNavContext(): NavContext {
	const ws = workspace();
	const enteredPath: Record<string, string[]> = {};
	const sel: Record<string, { nodes: string[]; edges: string[] }> = {};
	for (const p of collectPanels(ws.active.root)) {
		const path = pathToArray(asStateObject(p.state).subpatchPath);
		if (path.length) enteredPath[p.id] = path;
		const nodes = [...selection().nodes(p.id)];
		const edges = [...selection().edges(p.id)];
		if (nodes.length || edges.length) sel[p.id] = { nodes, edges };
	}
	return {
		activeWorkspaceId: ws.state.activeWorkspaceId,
		activePanelId: ws.activePanelId,
		enteredPath,
		selection: sel
	};
}

export async function restoreNavContext(ctx: NavContext): Promise<void> {
	const ws = workspace();
	if (ctx.activeWorkspaceId && ctx.activeWorkspaceId !== ws.state.activeWorkspaceId) {
		ws.selectTab(ctx.activeWorkspaceId);
	}
	// Drive each editor's sub-patch depth back via its persisted path; the panel
	// component reacts to subpatchPath and navigates there.
	for (const [panelId, path] of Object.entries(ctx.enteredPath)) {
		const p = findPanel(ws.active.root, panelId);
		if (!p) continue;
		const want = arrayToPath(path);
		if (asStateObject(p.state).subpatchPath !== want) {
			ws.setPanelState(panelId, { ...asStateObject(p.state), subpatchPath: want });
		}
	}
	// Restore the selection so the undone/redone change is highlighted.
	for (const [panelId, s] of Object.entries(ctx.selection)) {
		selection().setSelection(panelId, s.nodes, s.edges);
	}
	if (ctx.activePanelId) {
		ws.setActive(ctx.activePanelId);
		selection().setActiveEditor(ctx.activePanelId);
	}
}
