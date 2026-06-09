/**
 * Agent command surface — the flat, typed entry point for driving goofi-pipe
 * programmatically (a future in-app AI agent panel, Playwright e2e, the dev
 * console). Paired with `query` for reads.
 *
 * Every method delegates to the same store / editor logic the UI uses, so an
 * agent action and a human click take identical code paths — no behavior is
 * defined here, only a stable, discoverable façade. New verbs should route a
 * scattered call-site through here rather than reaching into a component.
 */
import { graph } from '$lib/stores/graph.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { ui } from '$lib/stores/ui.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { editorFor } from '$lib/panels/editorCommands';
import { setViewerKind } from '$lib/viewers/viewerState.svelte';
import { setViewerSetting } from '$lib/viewers/viewerSettings.svelte';
import type { ViewerKind } from '$lib/viewers/kind';
import type { LinkInfo } from '$lib/api/control';

/** The editor panel that viewport/selection verbs default to. */
function activeEditor(): string | null {
	return workspace().activePanelId;
}

export const commands = {
	// --- graph mutations ---------------------------------------------------
	addNode: (type: string, category: string, pos: [number, number] = [0, 0]): Promise<string> =>
		graph().addNode(type, category, pos),
	removeNode: (name: string): Promise<void> => graph().removeNode(name),
	removeNodes: (names: string[]): Promise<void> => graph().removeNodes(names),
	addLink: (link: LinkInfo): Promise<void> => graph().addLink(link),
	removeLink: (link: LinkInfo): Promise<void> => graph().removeLink(link),
	updateParam: (node: string, group: string, name: string, value: unknown): Promise<void> =>
		graph().updateParam(node, group, name, value),
	setExpression: (
		node: string,
		group: string,
		name: string,
		expression: string | null,
		opts?: { enabled?: boolean; triggers_process?: boolean; autoeval?: boolean }
	): Promise<void> => graph().setExpression(node, group, name, expression, opts),
	setNodePos: (name: string, pos: [number, number]): Promise<void> => graph().setNodePos(name, pos),
	cloneNodes: (names: string[], offset: [number, number] = [40, 40]): Promise<Record<string, string>> =>
		graph().cloneNodes(names, offset),

	// --- patch persistence -------------------------------------------------
	save: (path?: string): Promise<{ path: string; yaml: string }> =>
		graph().save(path, true, workspace().serialize()),
	loadText: (content: string): Promise<void> => graph().loadText(content),

	// --- selection / focus -------------------------------------------------
	select: (names: string[], panelId: string | null = activeEditor()): void => {
		if (panelId) selection().selectNodes(panelId, names);
	},
	clearSelection: (panelId: string | null = activeEditor()): void => {
		if (panelId) selection().clear(panelId);
	},
	focusNode: (name: string, panelId: string | null = activeEditor()): void =>
		editorFor(panelId)?.focusNode(name),

	// --- editor viewport ---------------------------------------------------
	openAddMenu: (panelId: string | null = activeEditor()): void => editorFor(panelId)?.openAddMenu(),
	fitView: (panelId: string | null = activeEditor()): void => editorFor(panelId)?.fitView(),

	// --- viewers -----------------------------------------------------------
	// Each mutator persists the slot's view state (debounced) so an agent-driven
	// change round-trips into the .gfi the same way a click does, regardless of
	// whether a canvas SlotViewer happens to be mounted for the slot.
	setSlotExpanded: (node: string, slot: string, expanded: boolean): void => {
		ui().setSlotExpanded(node, slot, expanded);
		graph().pushNodeViewers(node);
	},
	setViewerKind: (node: string, slot: string, kind: ViewerKind): void => {
		setViewerKind(node, slot, kind);
		graph().pushNodeViewers(node);
	},
	setViewerSetting: (node: string, slot: string, key: string, value: boolean | number | string): void => {
		setViewerSetting(node, slot, key, value);
		graph().pushNodeViewers(node);
	},

	// --- panels / layout ---------------------------------------------------
	bindNodeToPanel: (panelId: string, node: string): void => workspace().linkNodeToPanel(panelId, node),
	setPanelType: (panelId: string, type: string): void => workspace().setType(panelId, type),
	addTab: (panelType?: string): void => workspace().addTab(panelType)
};

export type Commands = typeof commands;
