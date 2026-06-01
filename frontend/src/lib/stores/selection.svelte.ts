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

function readInspectorPref(): boolean {
	try {
		return localStorage.getItem('goofi.inspectorOn') !== '0';
	} catch {
		return true;
	}
}

class SelectionStore {
	/** Per-editor-panel selection, keyed by panel id. */
	private map = $state<Record<string, PanelSel>>({});
	/** Last-focused editor panel — the standalone panels follow this one. */
	activeEditorId = $state<string | null>(null);
	/** Whether each editor's inspector overlay is shown (global toggle). */
	inspectorEnabled = $state(readInspectorPref());

	private sel(panelId: string | null): PanelSel {
		return (panelId && this.map[panelId]) || EMPTY;
	}
	private write(panelId: string, next: PanelSel): void {
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
		return graph().nodes.find((n) => n.name === name) ?? null;
	}

	/** Selected node of the last-focused editor — for standalone panels. */
	get activeSelectedNode(): NodeInstanceInfo | null {
		return this.selectedNode(this.activeEditorId);
	}

	setActiveEditor(panelId: string): void {
		if (this.activeEditorId !== panelId) this.activeEditorId = panelId;
	}

	toggleInspector(): void {
		this.inspectorEnabled = !this.inspectorEnabled;
		try {
			localStorage.setItem('goofi.inspectorOn', this.inspectorEnabled ? '1' : '0');
		} catch {
			/* best-effort */
		}
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

	clearNodes(panelId: string): void {
		const cur = this.sel(panelId);
		if (cur.nodes.size) this.write(panelId, { nodes: new Set(), edges: cur.edges });
	}
	clearEdges(panelId: string): void {
		const cur = this.sel(panelId);
		if (cur.edges.size) this.write(panelId, { nodes: cur.nodes, edges: new Set() });
	}
	clear(panelId: string): void {
		const cur = this.sel(panelId);
		if (cur.nodes.size || cur.edges.size) this.write(panelId, { nodes: new Set(), edges: new Set() });
	}

	/** Drop a closed panel's selection so the map doesn't accumulate. */
	forgetPanel(panelId: string): void {
		if (!this.map[panelId]) return;
		const { [panelId]: _drop, ...rest } = this.map;
		this.map = rest;
		if (this.activeEditorId === panelId) this.activeEditorId = null;
	}
}

let _sel: SelectionStore | null = null;
export function selection(): SelectionStore {
	if (!_sel) _sel = new SelectionStore();
	return _sel;
}
