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
import { workspace } from '$lib/workspace/workspace.svelte';
import { history } from '$lib/stores/history.svelte';
import { latestFrame } from '$lib/api/frames';
import { collectPanels } from '$lib/workspace/model';
import { asStateObject, linkedNodeName } from '$lib/workspace/panelState';
import { isArrayFrame, isStringFrame, type DataFrame } from '$lib/codec/decode';
import { reconstructMeta } from '$lib/editor/metaFormat';
import type { LinkInfo, NodeInstanceInfo, NodeTypeInfo } from '$lib/api/control';

export interface FrameSummary {
	dtype: string;
	shape?: number[];
	numeric?: { min: number; max: number; mean: number };
	text?: string;
}

/** A compact, DOM-free description of the latest frame on a slot — enough for an
 * agent to reason about live data without a vision pass. */
function summarize(frame: DataFrame | null): FrameSummary | null {
	if (!frame) return null;
	if (isArrayFrame(frame)) {
		const a = frame.data;
		let min = Infinity;
		let max = -Infinity;
		let sum = 0;
		let n = 0;
		const v = a.values;
		for (let i = 0; i < v.length; i++) {
			const x = Number(v[i]);
			if (!Number.isFinite(x)) continue;
			if (x < min) min = x;
			if (x > max) max = x;
			sum += x;
			n += 1;
		}
		// Report the node's TRUE shape, not the reduced wire shape (Option C): a
		// node-reduced frame carries meta.reduced with each axis's orig_len. Mirror
		// the inspector (reconstructMeta) so an agent reasons about real dimensions.
		const recon = reconstructMeta(frame.meta);
		const shape = Array.isArray(recon.shape) ? (recon.shape as number[]) : a.shape;
		return { dtype: a.dtype, shape, numeric: n ? { min, max, mean: sum / n } : undefined };
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
	node: (name: string): NodeInstanceInfo | null => graph().nodeByName(name),
	nodeParams: (name: string): NodeInstanceInfo['params'] | null =>
		graph().nodeByName(name)?.params ?? null,
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

	// --- history -----------------------------------------------------------
	canUndo: (): boolean => history().canUndo,
	canRedo: (): boolean => history().canRedo,
	undoLabel: (): string | null => history().undoLabel,
	redoLabel: (): string | null => history().redoLabel,
	historyLength: (): number => history().length
};

export type Query = typeof query;
