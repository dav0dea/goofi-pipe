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
import { history } from '$lib/stores/history.svelte';
import { setInlineKind, setInlineSetting, rawInlineView } from '$lib/viewers/inlineView.svelte';
import { recordViewChange } from '$lib/viewers/viewExecutors';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
import type { ViewerKind } from '$lib/viewers/kind';
import type { LinkInfo } from '$lib/api/control';
import type { GlobalType } from '$lib/crdt/graphDoc';

/** Raw (pre-resolution) inline view snapshot, for undo capture. */
function inlineSnap(node: string, slot: string): { kind?: ViewerKind; settings: SettingsMap } {
	const v = rawInlineView(node, slot);
	return { kind: v.kind, settings: { ...v.settings } };
}

/** The editor panel that viewport/selection verbs default to. */
function activeEditor(): string | null {
	return workspace().activePanelId;
}

export const commands = {
	// --- graph mutations ---------------------------------------------------
	addNode: (
		type: string,
		category: string,
		pos: [number, number] = [0, 0],
		instId?: string
	): Promise<string> => graph().addNode(type, category, pos, instId),
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
		opts?: { enabled?: boolean; triggers_process?: boolean }
	): Promise<void> => graph().setExpression(node, group, name, expression, opts),
	setNodePos: (name: string, pos: [number, number]): Promise<void> => graph().setNodePos(name, pos),
	cloneNodes: (names: string[], offset: [number, number] = [40, 40]): Promise<Record<string, string>> =>
		graph().cloneNodes(names, offset),

	// --- sub-patches -------------------------------------------------------
	groupNodes: (names: string[], pos: [number, number] = [0, 0]): Promise<string> =>
		graph().groupNodes(names, pos),
	expandInstance: (instId: string): Promise<void> => graph().expandInstance(instId),
	addBoundary: (
		instId: string,
		dir: 'in' | 'out',
		dtype: string,
		pos: [number, number] = [0, 0]
	): Promise<string> => graph().addBoundary(instId, dir, dtype, pos),
	wireBoundary: (
		instId: string,
		bndId: string,
		innerNode: string | null,
		innerSlot: string | null
	): Promise<void> => graph().wireBoundary(instId, bndId, innerNode, innerSlot),
	removeBoundary: (instId: string, bndId: string): Promise<void> =>
		graph().removeBoundary(instId, bndId),
	// Relabels the portal WITHOUT re-keying it: the exposed slot is still addressed by `bndId`, so
	// external wires survive and only what the user reads changes. That split is exactly what the
	// slot pickers have to honour — label out, boundary id in.
	renameBoundary: (instId: string, bndId: string, name: string): Promise<void> =>
		graph().renameBoundary(instId, bndId, name),

	// --- globals -----------------------------------------------------------
	// Patch-scoped named scalars (EditGlobal command ops, same path as the Globals panel — undoable).
	// Each REJECTS on a server refusal (invalid name / collision / a protected system global).
	// Expressions read them as `globals.<name>`.
	addGlobal: (name: string, value: number | string | boolean, type: GlobalType): Promise<void> =>
		graph().addGlobal(name, value, type),
	setGlobalValue: (name: string, value: number | string | boolean): Promise<void> =>
		graph().setGlobalValue(name, value),
	removeGlobal: (name: string): Promise<void> => graph().removeGlobal(name),
	renameGlobal: (oldName: string, newName: string): Promise<void> =>
		graph().renameGlobal(oldName, newName),

	// --- patch persistence -------------------------------------------------
	// The header's Save, whole: the file is written AND the patch is named, so
	// `query.graph().savePath` reads it back. The path is required — the no-path serialize-only
	// form went with "Save in browser" (removed 2026-08-08).
	save: (path: string): Promise<{ path: string }> => graph().save(path),
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
		const before = inlineSnap(node, slot);
		setInlineKind(node, slot, kind);
		graph().pushNodeViewers(node);
		recordViewChange({ kind: 'inline', node, slot }, before, inlineSnap(node, slot), `Viewer → ${kind}`);
	},
	setViewerSetting: (node: string, slot: string, key: string, value: boolean | number | string): void => {
		const before = inlineSnap(node, slot);
		setInlineSetting(node, slot, key, value);
		graph().pushNodeViewers(node);
		recordViewChange({ kind: 'inline', node, slot }, before, inlineSnap(node, slot), `Viewer ${key}`);
	},

	// --- panels / layout ---------------------------------------------------
	bindNodeToPanel: (panelId: string, node: string): void => workspace().linkNodeToPanel(panelId, node),
	setPanelType: (panelId: string, type: string): void => workspace().setType(panelId, type),
	addTab: (panelType?: string): void => workspace().addTab(panelType),

	// --- history -----------------------------------------------------------
	undo: (): Promise<void> => history().undo(),
	redo: (): Promise<void> => history().redo()
};

export type Commands = typeof commands;
