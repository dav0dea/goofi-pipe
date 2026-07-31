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
import { arrayToPath, asStateObject, pathToArray } from './panelState';
import type { NavContext } from '$lib/stores/history.svelte';

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
	const root = ws.active.root;
	// Drive each surviving editor's sub-patch depth back via its persisted path;
	// the panel component reacts to subpatchPath and navigates there.
	for (const [panelId, path] of Object.entries(ctx.enteredPath)) {
		const p = findPanel(root, panelId);
		if (!p) continue;
		const want = arrayToPath(path);
		if (asStateObject(p.state).subpatchPath !== want) {
			// Re-orienting the editor is navigation — the undone command dirties the patch on its
			// own merits, and a nav restore that undoes nothing must leave the flag alone.
			ws.setPanelState(panelId, { ...asStateObject(p.state), subpatchPath: want }, 'navigation');
		}
	}
	// Restore the selection on every panel that still exists.
	for (const [panelId, s] of Object.entries(ctx.selection)) {
		if (findPanel(root, panelId)) selection().setSelection(panelId, s.nodes, s.edges);
	}
	// Reorient the active panel. If the recorded panel was closed by the very
	// change we're undoing, fall back to a live node-editor panel so the restore
	// lands somewhere visible instead of silently no-op'ing — and replay the
	// recorded selection there so the change is still highlighted.
	let target = ctx.activePanelId;
	if (!target || !findPanel(root, target)) {
		const fallback = collectPanels(root).find((p) => p.panelType === 'node-editor') ?? collectPanels(root)[0];
		target = fallback?.id ?? null;
		const primary = ctx.activePanelId ? ctx.selection[ctx.activePanelId] : undefined;
		if (target && primary) selection().setSelection(target, primary.nodes, primary.edges);
	}
	if (target) {
		ws.setActive(target);
		selection().setActiveEditor(target);
	}
}
