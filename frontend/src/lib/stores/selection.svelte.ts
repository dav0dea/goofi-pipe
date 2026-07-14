/**
 * Graph selection — keyed per editor panel.
 *
 * Each node-editor panel has its own independent selection (selecting a node
 * in one panel doesn't affect another). The per-editor inspector reads its own
 * panel's selection; the standalone Parameters / Metadata / Errors panels
 * follow `activeSelectedNode` — the selection of whichever editor was last
 * focused.
 *
 * Selections are replaced (never mutated in place) so a plain assignment drives
 * reactivity. The click semantics are preserved from the original Editor,
 * including the asymmetry that a plain node click leaves edge selection intact
 * while a plain edge click clears node selection.
 */
import { graph } from './graph.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

interface PanelSel {
	nodes: Set<string>;
	edges: Set<string>;
}
const EMPTY: PanelSel = { nodes: new Set(), edges: new Set() };

function setEq(a: Set<string>, b: Set<string>): boolean {
	if (a.size !== b.size) return false;
	for (const x of a) if (!b.has(x)) return false;
	return true;
}

class SelectionStore {
	/** Per-editor-panel selection, keyed by panel id. */
	private map = $state<Record<string, PanelSel>>({});
	/** Last-focused editor panel — the standalone panels follow this one. */
	activeEditorId = $state<string | null>(null);
	/** Per-editor inspector visibility, keyed by panel id. Absent = enabled (the
	 * default). Ephemeral — deliberately not persisted to any browser storage. */
	private inspectorOn = $state<Record<string, boolean>>({});

	private sel(panelId: string | null): PanelSel {
		return (panelId && this.map[panelId]) || EMPTY;
	}
	private write(panelId: string, next: PanelSel): void {
		// Skip a no-op write: re-selecting the already-selected node (e.g. a drag-start
		// mousedown on a selected node) would otherwise allocate a fresh selection object,
		// retriggering the editor's flowNodes effect mid-drag so Svelte Flow's
		// onnodedragstart never fires and the node can't be dragged (e.g. into a panel).
		const cur = this.map[panelId];
		if (cur && setEq(cur.nodes, next.nodes) && setEq(cur.edges, next.edges)) return;
		this.map = { ...this.map, [panelId]: next };
	}

	nodes(panelId: string | null): Set<string> {
		return this.sel(panelId).nodes;
	}
	edges(panelId: string | null): Set<string> {
		return this.sel(panelId).edges;
	}

	/** The single selected node of `panelId`, or null for zero / many. */
	selectedNode(panelId: string | null): NodeInstanceInfo | null {
		const ns = this.sel(panelId).nodes;
		if (ns.size !== 1) return null;
		const name = [...ns][0];
		return graph().nodeById(name);
	}

	/** Selected node of the last-focused editor — for standalone panels. */
	get activeSelectedNode(): NodeInstanceInfo | null {
		return this.selectedNode(this.activeEditorId);
	}

	setActiveEditor(panelId: string): void {
		if (this.activeEditorId !== panelId) this.activeEditorId = panelId;
	}

	/** Whether `panelId`'s inspector pane appears when it has a node selected.
	 * Defaults to enabled; a null panel id (no active editor) reads as off. */
	inspectorEnabledFor(panelId: string | null): boolean {
		return panelId !== null ? (this.inspectorOn[panelId] ?? true) : false;
	}
	toggleInspectorFor(panelId: string): void {
		this.inspectorOn = { ...this.inspectorOn, [panelId]: !this.inspectorEnabledFor(panelId) };
	}

	// --- node selection ----------------------------------------------------

	clickNode(panelId: string, name: string, additive: boolean): void {
		const cur = this.sel(panelId);
		if (additive) {
			const nodes = new Set(cur.nodes);
			if (nodes.has(name)) nodes.delete(name);
			else nodes.add(name);
			this.write(panelId, { nodes, edges: cur.edges });
		} else {
			this.write(panelId, { nodes: new Set([name]), edges: cur.edges });
		}
	}

	selectNodes(panelId: string, names: Iterable<string>): void {
		this.write(panelId, { nodes: new Set(names), edges: this.sel(panelId).edges });
	}

	/** Replace both node and edge selection at once (NavContext restore). */
	setSelection(panelId: string, nodes: Iterable<string>, edges: Iterable<string>): void {
		this.write(panelId, { nodes: new Set(nodes), edges: new Set(edges) });
	}

	// --- edge selection ----------------------------------------------------

	clickEdge(panelId: string, id: string, additive: boolean): void {
		const cur = this.sel(panelId);
		if (additive) {
			const edges = new Set(cur.edges);
			if (edges.has(id)) edges.delete(id);
			else edges.add(id);
			this.write(panelId, { nodes: cur.nodes, edges });
		} else {
			this.write(panelId, { nodes: new Set(), edges: new Set([id]) });
		}
	}

	// --- clearing ----------------------------------------------------------

	clickPane(panelId: string, shift: boolean): void {
		if (shift) return;
		this.clear(panelId);
	}

	clear(panelId: string): void {
		const cur = this.sel(panelId);
		if (cur.nodes.size || cur.edges.size) this.write(panelId, { nodes: new Set(), edges: new Set() });
	}

	/** Drop ALL per-panel state — selection and inspector visibility. Called when
	 * the layout is wholesale-replaced (reset / hydrate): a loaded `.gfi` keeps
	 * its saved panel ids, which can collide with ids used earlier this session,
	 * so any leftover per-id state would silently apply to the new panels. */
	forgetAll(): void {
		this.map = {};
		this.inspectorOn = {};
		this.activeEditorId = null;
	}
}

let _sel: SelectionStore | null = null;
export function selection(): SelectionStore {
	if (!_sel) _sel = new SelectionStore();
	return _sel;
}
