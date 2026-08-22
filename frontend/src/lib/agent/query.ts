/** Agent read/introspection surface — paired with `commands`. */
import { graph } from '$lib/stores/graph.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { ui } from '$lib/stores/ui.svelte';
import { workspace } from 'panelty';
import { history } from '$lib/stores/history.svelte';
import { latestFrame } from '$lib/api/frames';
import { collectPanels } from 'panelty';
import { asStateObject, linkedNodeName } from 'panelty';
import { isArrayFrame, isStringFrame, type DataFrame } from '$lib/codec/decode';
import { reconstructMeta } from '$lib/editor/metaFormat';
import { summaryOf } from '$lib/viewers/viewMeta';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { InstanceInfo, LinkInfo, NodeInstanceInfo, NodeTypeInfo } from '$lib/api/control';
import type { GlobalView } from '$lib/crdt/graphDoc';

export interface FrameSummary {
	dtype: string;
	shape?: number[];
	/** Element count of the reduced wire array `numeric` covers; `shape` stays the node's TRUE original. */
	reducedLength?: number;
	numeric?: { min: number; max: number; mean: number };
	text?: string;
}

/** A compact, DOM-free description of the latest frame on a slot. */
function summarize(frame: DataFrame | null): FrameSummary | null {
	if (!frame) return null;
	if (isArrayFrame(frame)) {
		const a = frame.data;
		const s = summaryOf(a);
		const recon = reconstructMeta(frame.meta);
		const shape = Array.isArray(recon.shape) ? (recon.shape as number[]) : a.shape;
		const reduced = !!frame.meta && typeof frame.meta === 'object' && 'reduced' in frame.meta;
		return {
			dtype: a.dtype,
			shape,
			numeric: s.min !== null ? { min: s.min, max: s.max as number, mean: s.mean as number } : undefined,
			...(reduced ? { reducedLength: a.values.length } : {})
		};
	}
	if (isStringFrame(frame)) return { dtype: 'STRING', text: frame.data };
	return { dtype: frame.dtype };
}

export interface PanelView {
	panelId: string;
	type: string;
	node: string | null;
	slot: string | null;
	kind: string | null;
}

export const query = {
	graph: (): {
		nodes: NodeInstanceInfo[];
		links: LinkInfo[];
		savePath: string | null;
		unsavedChanges: boolean;
	} => {
		const g = graph();
		return {
			nodes: g.nodes,
			links: g.links,
			savePath: g.savePath,
			unsavedChanges: g.unsavedChanges
		};
	},
	nodeTypes: (): NodeTypeInfo[] | null => graph().nodeTypes,
	/** Whether the replica has pulled from the manager yet; until true, `graph()` reads describe an EMPTY replica. */
	docSynced: (): boolean => graph().docSynced,
	/** Every patch global (system + user), in system-first/creation order. */
	globals: (): GlobalView[] => graph().globals,
	/** Every sub-patch instance by uid. ROOT is excluded — it is the canvas, not a sub-patch. */
	instances: (): Record<string, InstanceInfo> => {
		const { [ROOT_ID]: _root, ...rest } = graph().instances;
		return rest;
	},
	/** One sub-patch instance by uid, or null (ROOT is not a sub-patch). */
	instance: (uid: string): InstanceInfo | null =>
		uid === ROOT_ID ? null : graph().instances[uid] ?? null,
	node: (uid: string): NodeInstanceInfo | null => graph().nodeById(uid),
	nodeParams: (uid: string): NodeInstanceInfo['params'] | null =>
		graph().nodeById(uid)?.params ?? null,
	selection: (
		panelId: string | null = workspace().activePanelId
	): { nodes: string[]; edges: string[] } => {
		const sel = selection();
		return { nodes: [...sel.nodes(panelId)], edges: [...sel.edges(panelId)] };
	},
	activeNode: (): NodeInstanceInfo | null => selection().activeSelectedNode,
	latestFrame: (node: string, slot: string): DataFrame | null => latestFrame(node, slot),
	frameSummary: (node: string, slot: string): FrameSummary | null =>
		summarize(latestFrame(node, slot)),
	panels: (): PanelView[] =>
		collectPanels(workspace().root).map((p) => {
			const s = asStateObject(p.state);
			return {
				panelId: p.id,
				type: p.panelType,
				node: linkedNodeName(p.state),
				slot: typeof s.slot === 'string' ? s.slot : null,
				kind: typeof s.kind === 'string' ? s.kind : null
			};
		}),

	/** True while an in-panel editor holds the keyboard, so global undo/redo stands down. */
	modalOpen: (): boolean => ui().modalOpen,

	canUndo: (): boolean => history().canUndo,
	canRedo: (): boolean => history().canRedo,
	undoLabel: (): string | null => history().undoLabel,
	redoLabel: (): string | null => history().redoLabel,
	historyLength: (): number => history().length
};

export type Query = typeof query;
