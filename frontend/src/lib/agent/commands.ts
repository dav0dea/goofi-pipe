/** Flat, typed command facade over the store logic the UI uses; paired with `query` for reads. */
import { graph } from '$lib/stores/graph.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { workspace } from 'panelty';
import { editorFor } from '$lib/panels/editorCommands';
import { history } from '$lib/stores/history.svelte';
import { slotView } from '$lib/viewers/inlineView';
import { recordViewChange } from '$lib/viewers/viewExecutors';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
import type { ViewerKind } from '$lib/viewers/kind';
import type { LinkInfo, ScanDiff } from '$lib/api/control';
import type { GlobalType } from '$lib/crdt/graphDoc';

/** Raw (pre-resolution) inline view snapshot, for undo capture. */
function inlineSnap(node: string, slot: string): { kind?: ViewerKind; settings: SettingsMap } {
	const v = slotView(graph().nodeById(node), slot);
	return { kind: v.kind, settings: { ...v.settings } };
}

/** The editor panel that viewport/selection verbs default to. */
function activeEditor(): string | null {
	return workspace().activePanelId;
}

export const commands = {
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
	renameNode: (uid: string, name: string): Promise<void> => graph().renameNode(uid, name),
	cloneNodes: (names: string[], offset: [number, number] = [40, 40]): Promise<Record<string, string>> =>
		graph().cloneNodes(names, offset),

	groupNodes: (names: string[], pos: [number, number] = [0, 0]): Promise<string> =>
		graph().groupNodes(names, pos),
	expandInstance: (instId: string): Promise<void> => graph().expandInstance(instId),

	addGlobal: (name: string, value: number | string | boolean, type: GlobalType): Promise<void> =>
		graph().addGlobal(name, value, type),
	setGlobalValue: (name: string, value: number | string | boolean): Promise<void> =>
		graph().setGlobalValue(name, value),
	removeGlobal: (name: string): Promise<void> => graph().removeGlobal(name),
	renameGlobal: (oldName: string, newName: string): Promise<void> =>
		graph().renameGlobal(oldName, newName),

	save: (path: string): Promise<{ path: string }> => graph().save(path),
	load: (path: string): Promise<void> => graph().load(path),
	newPatch: (): Promise<void> => graph().newPatch(),
	// Returns the patch workspace directory — a per-run temp path only the manager knows.
	openWorkspace: (): Promise<string> => graph().openWorkspace(),
	rescanNodes: (): Promise<ScanDiff> => graph().rescanNodes(),

	select: (names: string[], panelId: string | null = activeEditor()): void => {
		if (panelId) selection().selectNodes(panelId, names);
	},
	clearSelection: (panelId: string | null = activeEditor()): void => {
		if (panelId) selection().clear(panelId);
	},
	focusNode: (name: string, panelId: string | null = activeEditor()): void =>
		editorFor(panelId)?.focusNode(name),

	openAddMenu: (panelId: string | null = activeEditor()): void => editorFor(panelId)?.openAddMenu(),
	fitView: (panelId: string | null = activeEditor()): void => editorFor(panelId)?.fitView(),

	setSlotExpanded: (node: string, slot: string, expanded: boolean): void =>
		graph().setSlotView(node, slot, { collapsed: !expanded }),
	setViewerKind: (node: string, slot: string, kind: ViewerKind): void => {
		const before = inlineSnap(node, slot);
		const after = { ...before, kind };
		graph().setSlotView(node, slot, after);
		recordViewChange({ kind: 'inline', node, slot }, before, after, `Viewer → ${kind}`);
	},
	setViewerSetting: (node: string, slot: string, key: string, value: boolean | number | string): void => {
		const before = inlineSnap(node, slot);
		const after = { kind: before.kind, settings: { ...before.settings, [key]: value } };
		graph().setSlotView(node, slot, after);
		recordViewChange({ kind: 'inline', node, slot }, before, after, `Viewer ${key}`);
	},

	bindNodeToPanel: (panelId: string, node: string): void => workspace().linkNodeToPanel(panelId, node),
	setPanelType: (panelId: string, type: string): void => workspace().setType(panelId, type),
	addTab: (panelType?: string): void => workspace().addTab(panelType),

	undo: (): Promise<void> => history().undo(),
	redo: (): Promise<void> => history().redo()
};

export type Commands = typeof commands;
