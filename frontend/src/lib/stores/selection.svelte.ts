/** Graph selection, keyed per editor panel; a selection is replaced, never mutated in place. */
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
	/** Per-editor inspector visibility, keyed by panel id. Absent = enabled. */
	private inspectorOn = $state<Record<string, boolean>>({});
	/** Per-editor TRANSIENT dismissal (the ✕), cleared in `write()` — the one choke-point every
	 * real selection change funnels through. `inspectorOn` is the standing preference (the ◧). */
	private inspectorDismissed = $state<Record<string, boolean>>({});
	/** While on, a plain click adds to the selection — the coarse-pointer stand-in for
	 * shift/ctrl/meta. Session-wide, so `forgetAll` leaves it. */
	multiSelect = $state(false);

	toggleMultiSelect(): void {
		this.multiSelect = !this.multiSelect;
	}

	private sel(panelId: string | null): PanelSel {
		return (panelId && this.map[panelId]) || EMPTY;
	}
	private write(panelId: string, next: PanelSel): void {
		// A no-op write would allocate a fresh selection object, retriggering the editor's flowNodes
		// effect mid-drag so Svelte Flow's onnodedragstart never fires.
		const cur = this.map[panelId];
		if (cur && setEq(cur.nodes, next.nodes) && setEq(cur.edges, next.edges)) return;
		this.map = { ...this.map, [panelId]: next };
		// A real selection change re-arms a dismissed inspector — here and nowhere else.
		if (this.inspectorDismissed[panelId]) {
			const { [panelId]: _, ...rest } = this.inspectorDismissed;
			this.inspectorDismissed = rest;
		}
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

	/** Whether `panelId`'s inspector pane appears; a null panel id reads as off. */
	inspectorEnabledFor(panelId: string | null): boolean {
		return panelId !== null ? (this.inspectorOn[panelId] ?? true) : false;
	}
	toggleInspectorFor(panelId: string): void {
		this.inspectorOn = { ...this.inspectorOn, [panelId]: !this.inspectorEnabledFor(panelId) };
	}
	/** Close the pane until the selection next changes. The ✕'s verb — never the ◧'s. */
	dismissInspectorFor(panelId: string): void {
		this.inspectorDismissed = { ...this.inspectorDismissed, [panelId]: true };
	}
	/** Bring the pane back regardless of how it was hidden — the ◧'s "show" half. */
	showInspectorFor(panelId: string): void {
		this.inspectorOn = { ...this.inspectorOn, [panelId]: true };
		if (this.inspectorDismissed[panelId]) {
			const { [panelId]: _, ...rest } = this.inspectorDismissed;
			this.inspectorDismissed = rest;
		}
	}
	/** What the pane actually renders from: the standing preference minus a live dismissal. */
	inspectorVisibleFor(panelId: string | null): boolean {
		return (
			this.inspectorEnabledFor(panelId) &&
			!(panelId !== null && (this.inspectorDismissed[panelId] ?? false))
		);
	}

	/** A click adds rather than replaces on a modifier OR while multi-select mode is on; folded in
	 * here, not at each call site, so no caller can forget the mode. */
	clickNode(panelId: string, name: string, modifier: boolean): void {
		const cur = this.sel(panelId);
		const additive = modifier || this.multiSelect;
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

	/** Same fold as `clickNode`; the mode covers edges too, since a plain edge click clears nodes. */
	clickEdge(panelId: string, id: string, modifier: boolean): void {
		const cur = this.sel(panelId);
		const additive = modifier || this.multiSelect;
		if (additive) {
			const edges = new Set(cur.edges);
			if (edges.has(id)) edges.delete(id);
			else edges.add(id);
			this.write(panelId, { nodes: cur.nodes, edges });
		} else {
			this.write(panelId, { nodes: new Set(), edges: new Set([id]) });
		}
	}

	/** The same fold, on empty canvas: with the mode on, a stray tap must not wipe the selection. */
	clickPane(panelId: string, shift: boolean): void {
		if (shift || this.multiSelect) return;
		this.clear(panelId);
	}

	clear(panelId: string): void {
		const cur = this.sel(panelId);
		if (cur.nodes.size || cur.edges.size) this.write(panelId, { nodes: new Set(), edges: new Set() });
	}

	/** Drop ALL per-panel state on a layout replace: a loaded `.gfi` keeps its saved panel ids,
	 * which can collide with ids this session already used. */
	forgetAll(): void {
		this.map = {};
		this.inspectorOn = {};
		this.inspectorDismissed = {};
		this.activeEditorId = null;
	}
}

let _sel: SelectionStore | null = null;
export function selection(): SelectionStore {
	if (!_sel) _sel = new SelectionStore();
	return _sel;
}
