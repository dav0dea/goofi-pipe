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
	/** Per-editor TRANSIENT dismissal — the ✕ in the pane's header. A close, not an off-switch:
	 * it holds only until the panel's selection actually CHANGES (deselect+reselect, or another
	 * node), which is why it clears inside `write()` — the one choke-point every real selection
	 * change funnels through. The same-node re-click that drag-start suppresses is a skipped
	 * write, so it leaves a dismissal standing by construction. `inspectorOn` above is the
	 * standing preference (the ◧ toggle); this map never outlives a selection change. */
	private inspectorDismissed = $state<Record<string, boolean>>({});
	/** While on, a plain click adds to the selection instead of replacing it — the coarse-pointer
	 * stand-in for shift/ctrl/meta, which a phone has none of (D-R4). A MODE, not a gesture: the
	 * user asked for something you switch on and see, not a second way to tap. Session-wide and
	 * deliberately not persisted (it is not the patch's business), so `forgetAll` leaves it. */
	multiSelect = $state(false);

	toggleMultiSelect(): void {
		this.multiSelect = !this.multiSelect;
	}

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
		// A REAL selection change re-arms a dismissed inspector (the ✕ is a close, not an
		// off-switch). Here and nowhere else: every genuine change funnels through this write,
		// and the no-op skip above keeps a drag-start's same-node re-click from reviving it.
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

	/** Whether `panelId`'s inspector pane appears when it has a node selected.
	 * Defaults to enabled; a null panel id (no active editor) reads as off. */
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
	/** Bring the pane back regardless of how it was hidden — the ◧'s "show" half: it clears a
	 * dismissal AND flips the standing preference on, so one press always answers "show it". */
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

	// --- node selection ----------------------------------------------------

	/** A click adds rather than replaces when the caller says so (shift/ctrl/meta) OR while
	 * multi-select mode is on. Folded in here, not OR-ed in at each call site, so "additive" has
	 * one definition and a caller cannot be written that forgets the mode. */
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

	// --- edge selection ----------------------------------------------------

	/** Same fold as `clickNode`. The mode covers edges too because the asymmetry it would
	 * otherwise leave is the surprising one: a plain edge click CLEARS the node selection, so a
	 * stray tap on a cable would undo the multi-node selection the mode exists to build. */
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

	// --- clearing ----------------------------------------------------------

	/** The same fold `clickNode`/`clickEdge` take, on the biggest target of all. A phone has no
	 * shift, so with the mode on a stray tap on empty canvas wiped the multi-node selection the
	 * mode exists to build. Deliberately shipped WITH the header menu's `Clear selection` row: this
	 * is touch's only clear-all door, since Escape is keyboard-only. */
	clickPane(panelId: string, shift: boolean): void {
		if (shift || this.multiSelect) return;
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
		this.inspectorDismissed = {};
		this.activeEditorId = null;
	}
}

let _sel: SelectionStore | null = null;
export function selection(): SelectionStore {
	if (!_sel) _sel = new SelectionStore();
	return _sel;
}
