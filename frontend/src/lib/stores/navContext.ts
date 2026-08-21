/** Capture and restore where an undo/redo happened: active tab, panel, sub-patch depth, selection. */
import { workspace } from 'panelty';
import { selection } from './selection.svelte';
import { collectPanels, findPanel } from 'panelty';
import { arrayToPath, asStateObject, pathToArray } from 'panelty';
import type { NavContext } from './history.svelte';

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
	for (const [panelId, path] of Object.entries(ctx.enteredPath)) {
		const p = findPanel(root, panelId);
		if (!p) continue;
		const want = arrayToPath(path);
		if (asStateObject(p.state).subpatchPath !== want) {
			// Re-orienting the editor is navigation, so it must not dirty the patch.
			ws.setPanelState(panelId, { ...asStateObject(p.state), subpatchPath: want }, 'navigation');
		}
	}
	for (const [panelId, s] of Object.entries(ctx.selection)) {
		if (findPanel(root, panelId)) selection().setSelection(panelId, s.nodes, s.edges);
	}
	// The change being undone can have closed the recorded panel; land somewhere visible instead.
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
