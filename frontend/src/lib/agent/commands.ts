/** Flat, typed command facade over the store logic the UI uses; paired with `query` for reads. */
import type { SourcePatch } from '$lib/api/types';
import { graph } from '$lib/stores/graph.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { workspace } from 'panelty';
import { history } from '$lib/stores/history.svelte';
import type { LinkInfo } from '$lib/api/control';
import type { ControlView } from '$lib/crdt/graphDoc';
import type { GlobalType, LockView } from '$lib/crdt/graphDoc';

/** The editor panel that viewport/selection verbs default to. */
function activeEditor(): string | null {
	return workspace().activePanelId;
}

export const commands = {
	addNode: (type: string, pos: [number, number] = [0, 0], instId?: string): Promise<string> =>
		graph().addNode(type, pos, instId),
	removeNode: (name: string): Promise<void> => graph().removeNode(name),
	removeNodes: (names: string[]): Promise<void> => graph().removeNodes(names),
	addLink: (link: LinkInfo): Promise<void> => graph().addLink(link),
	updateParam: (node: string, group: string, name: string, value: unknown): Promise<void> =>
		graph().updateParam(node, group, name, value),
	setSource: (node: string, group: string, name: string, source: SourcePatch): Promise<void> =>
		graph().setSource(node, group, name, source),
	setNodePos: (name: string, pos: [number, number]): Promise<void> => graph().setNodePos(name, pos),
	renameNode: (uid: string, name: string): Promise<void> => graph().renameNode(uid, name),

	groupNodes: (names: string[], pos: [number, number] = [0, 0]): Promise<string> =>
		graph().groupNodes(names, pos),
	expandInstance: (instId: string): Promise<void> => graph().expandInstance(instId),

	addGlobal: (
		name: string,
		value: number | string | boolean,
		type: GlobalType,
		control?: ControlView
	): Promise<void> => graph().addGlobal(name, value, type, control),
	removeGlobal: (name: string): Promise<void> => graph().removeGlobal(name),
	setGlobalControl: (name: string, control: ControlView): Promise<void> =>
		graph().setGlobalControl(name, control),
	renameGlobal: (from: string, to: string): Promise<void> => graph().renameGlobal(from, to),
	renameGlobalGroup: (from: string, to: string): Promise<void> => graph().renameGlobalGroup(from, to),
	lockGlobal: (name: string, lock: Partial<LockView>): Promise<void> => graph().lockGlobal(name, lock),
	lockGlobalGroup: (group: string, lock: Partial<LockView>): Promise<void> =>
		graph().lockGlobalGroup(group, lock),

	save: (path: string): Promise<{ path: string }> => graph().save(path),
	newPatch: (): Promise<void> => graph().newPatch(),

	select: (names: string[], panelId: string | null = activeEditor()): void => {
		if (panelId) selection().selectNodes(panelId, names);
	},
	clearSelection: (panelId: string | null = activeEditor()): void => {
		if (panelId) selection().clear(panelId);
	},
	bindNodeToPanel: (panelId: string, node: string): void => workspace().linkNodeToPanel(panelId, node),
	setPanelType: (panelId: string, type: string): void => workspace().setType(panelId, type),
	setPanelState: (panelId: string, state: Record<string, unknown>): void =>
		workspace().setPanelState(panelId, state),
	addTab: (panelType?: string): void => workspace().addTab(panelType),

	undo: (): Promise<void> => history().undo(),
	redo: (): Promise<void> => history().redo()
};

export type Commands = typeof commands;
