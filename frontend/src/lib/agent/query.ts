/**
 * Agent read/introspection surface — paired with `commands`.
 *
 * Answers the questions an agent (or a Playwright assertion) needs — what is the
 * graph, what's selected, what is each panel showing, what is a viewer's latest
 * data — without touching the DOM or walking the layout-tree internals. Reads go
 * straight through the same stores the UI binds to.
 */
import { graph } from '$lib/stores/graph.svelte';
import { selection } from '$lib/stores/selection.svelte';
import { ui } from '$lib/stores/ui.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import { history } from '$lib/stores/history.svelte';
import { latestFrame } from '$lib/api/frames';
import { collectPanels } from '$lib/workspace/model';
import { asStateObject, linkedNodeName } from '$lib/workspace/panelState';
import { isArrayFrame, isStringFrame, type DataFrame } from '$lib/codec/decode';
import { reconstructMeta } from '$lib/editor/metaFormat';
import { summaryOf } from '$lib/viewers/viewMeta';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import type { InstanceInfo, LinkInfo, NodeInstanceInfo, NodeTypeInfo } from '$lib/api/control';
import type { GlobalView } from '$lib/crdt/graphDoc';

export interface FrameSummary {
	dtype: string;
	shape?: number[];
	/** Present when the frame was node-reduced (Option C): the element count of the
	 * reduced wire array that `numeric` was computed over. `shape` is the node's TRUE
	 * original, so the two intentionally differ — and for an envelope reduction the
	 * mean is over interleaved per-bin min/max, not the original signal. */
	reducedLength?: number;
	numeric?: { min: number; max: number; mean: number };
	text?: string;
}

/** A compact, DOM-free description of the latest frame on a slot — enough for an
 * agent to reason about live data without a vision pass. */
function summarize(frame: DataFrame | null): FrameSummary | null {
	if (!frame) return null;
	if (isArrayFrame(frame)) {
		const a = frame.data;
		// Reuse the viewers' pure min/max/mean reduction (same skip-non-finite algorithm)
		// rather than re-implementing it here.
		const s = summaryOf(a);
		// Report the node's TRUE shape, not the reduced wire shape (Option C): a
		// node-reduced frame carries meta.reduced with each axis's orig_len. Mirror
		// the inspector (reconstructMeta) so an agent reasons about real dimensions.
		// `numeric` stays over the reduced wire array; surface its length (reducedLength)
		// when reduced so the shape/stats gap is explicit, not silent.
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
	/** Bound node for a linkable panel, else null. */
	node: string | null;
	/** Chosen slot for a Viewer/Metadata panel, else null. */
	slot: string | null;
	/** Chosen viewer kind for a Viewer panel, else null. */
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
	/** Every patch global (system + user), doc-authoritative, in system-first/creation order.
	 * Each is `{name, value, type, system}` — expressions read them as `globals.<name>`. */
	globals: (): GlobalView[] => graph().globals,
	/** Every sub-patch instance, keyed by uid (the universal node key). Each is the
	 * server-computed record (parent, members{local:{uid,is_instance}}, slots, siblings,
	 * error, …) the editor mirrors. ROOT (the materialized root scope) is excluded — it's
	 * the canvas, not a sub-patch the agent operates on. */
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
		collectPanels(workspace().active.root).map((p) => {
			const s = asStateObject(p.state);
			return {
				panelId: p.id,
				type: p.panelType,
				node: linkedNodeName(p.state),
				slot: typeof s.slot === 'string' ? s.slot : null,
				kind: typeof s.kind === 'string' ? s.kind : null
			};
		}),

	/** True while an in-panel editor (the fx multi-line) holds the keyboard, so global undo/redo
	 * stands down. Exposed so the e2e can assert the standdown lifted after a node switch. */
	modalOpen: (): boolean => ui().modalOpen,

	// --- history -----------------------------------------------------------
	canUndo: (): boolean => history().canUndo,
	canRedo: (): boolean => history().canRedo,
	undoLabel: (): string | null => history().undoLabel,
	redoLabel: (): string | null => history().redoLabel,
	historyLength: (): number => history().length
};

export type Query = typeof query;
