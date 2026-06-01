/**
 * Reactive workspace store — the only stateful layer of the panel system.
 *
 * Holds the `WorkspaceState` (tabs + their layout trees) plus ephemeral UI
 * state (which panel is active, which is maximized). All structural changes go
 * through the pure ops in `model.ts`; this layer just owns the `$state`, picks
 * sensible follow-up selection, and persists to localStorage.
 *
 * Persistence is explicit (a `_commit()` call at the end of each mutation)
 * rather than `$effect`-driven, because the store is a module-level singleton
 * created outside any component/effect context.
 */
import {
	closePanel,
	cloneWithNewIds,
	defaultWorkspaceState,
	DEFAULT_PANEL_TYPE,
	extractPanel,
	findPanel,
	firstPanelId,
	insertNodeAtPanel,
	makeWorkspace,
	reseedIds,
	resizeSplit,
	setPanelState,
	setPanelType,
	splitPanel,
	type Direction,
	type LayoutNode,
	type Workspace,
	type WorkspaceState
} from './model';
import { getPanelType } from './registry';

/** A drag in progress. A panel and a tab are both just a `LayoutNode` being
 * moved — the only difference is where it came from, which `_takeNode` knows
 * how to detach. */
export type DragRef =
	| { kind: 'panel'; workspaceId: string; panelId: string }
	| { kind: 'tab'; workspaceId: string };

function isValidState(s: unknown): s is WorkspaceState {
	if (typeof s !== 'object' || s === null) return false;
	const obj = s as Record<string, unknown>;
	if (!Array.isArray(obj.workspaces) || obj.workspaces.length === 0) return false;
	if (typeof obj.activeWorkspaceId !== 'string') return false;
	return obj.workspaces.some((w) => (w as Workspace).id === obj.activeWorkspaceId);
}

class WorkspaceStore {
	state = $state<WorkspaceState>(defaultWorkspaceState());
	/** Last-focused panel id — keyboard shortcuts scope to this. */
	activePanelId = $state<string | null>(null);
	/** When set, only this panel renders, filling the workspace. */
	maximizedPanelId = $state<string | null>(null);
	/** The panel or tab currently being dragged. While set, panels show edge
	 * drop zones and the tab bar accepts the drop, so the dragged node can be
	 * repositioned in the layout or turned into a tab. */
	dragging = $state<DragRef | null>(null);

	// The layout is *not* persisted to localStorage. It lives only in the
	// running patch (pushed to the manager) and the .gfi on save; a fresh
	// goofi-pipe with no patch therefore starts at the default layout.
	constructor() {
		this.activePanelId = firstPanelId(this.active.root);
	}

	get active(): Workspace {
		return (
			this.state.workspaces.find((w) => w.id === this.state.activeWorkspaceId) ??
			this.state.workspaces[0]
		);
	}

	/** Plain (de-proxied) snapshot for embedding in a saved `.gfi` patch and
	 * for pushing into the running patch. `$state.snapshot` unwraps Svelte's
	 * reactive proxy — `structuredClone` chokes on it, and the result must be a
	 * plain JSON object for the WS. */
	serialize(): WorkspaceState {
		return $state.snapshot(this.state) as WorkspaceState;
	}

	/** Apply a layout restored from a `.gfi` patch (or any external source). */
	hydrate(state: unknown): void {
		if (!isValidState(state)) return;
		reseedIds(state);
		this.state = state;
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(this.active.root);
	}

	private _setRoot(workspaceId: string, root: LayoutNode): void {
		this.state = {
			...this.state,
			workspaces: this.state.workspaces.map((w) =>
				w.id === workspaceId ? { ...w, root } : w
			)
		};
	}

	private _updateActiveRoot(fn: (root: LayoutNode) => LayoutNode | null): void {
		const ws = this.active;
		const root = fn(ws.root);
		if (!root || root === ws.root) return;
		this._setRoot(ws.id, root);
	}

	// --- layout mutations --------------------------------------------------

	/** Split a panel. `newType` defaults to the source panel's own type, so a
	 * split inherits content (Blender-style); the user then swaps if desired. */
	split(panelId: string, direction: Direction, placeBefore = false, newType?: string): void {
		const ws = this.active;
		const src = findPanel(ws.root, panelId);
		const type = newType ?? src?.panelType ?? DEFAULT_PANEL_TYPE;
		const { root, newPanelId } = splitPanel(ws.root, panelId, direction, placeBefore, type);
		if (root === ws.root) return;
		this._setRoot(ws.id, root);
		if (newPanelId) this.activePanelId = newPanelId;
	}

	close(panelId: string): void {
		const ws = this.active;
		const root = closePanel(ws.root, panelId);
		if (!root) return;
		this._setRoot(ws.id, root);
		if (this.maximizedPanelId === panelId) this.maximizedPanelId = null;
		if (this.activePanelId === panelId) this.activePanelId = firstPanelId(root);
	}

	resize(splitId: string, dividerIndex: number, delta: number): void {
		this._updateActiveRoot((root) => resizeSplit(root, splitId, dividerIndex, delta));
	}

	setType(panelId: string, panelType: string): void {
		const ds = getPanelType(panelType)?.defaultState?.();
		this._updateActiveRoot((root) => setPanelType(root, panelId, panelType, ds));
	}

	setPanelState(panelId: string, state: unknown): void {
		this._updateActiveRoot((root) => setPanelState(root, panelId, state));
	}

	setActive(panelId: string): void {
		if (this.activePanelId !== panelId) this.activePanelId = panelId;
	}

	toggleMaximize(panelId: string): void {
		this.maximizedPanelId = this.maximizedPanelId === panelId ? null : panelId;
	}

	// --- tabs --------------------------------------------------------------

	private _uniqueName(base: string): string {
		const names = new Set(this.state.workspaces.map((w) => w.name));
		if (!names.has(base)) return base;
		let i = 2;
		while (names.has(`${base} ${i}`)) i += 1;
		return `${base} ${i}`;
	}

	addTab(panelType: string = DEFAULT_PANEL_TYPE): void {
		const ws = makeWorkspace(this._uniqueName('Layout'), panelType);
		this.state = {
			workspaces: [...this.state.workspaces, ws],
			activeWorkspaceId: ws.id
		};
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(ws.root);	}

	selectTab(workspaceId: string): void {
		if (this.state.activeWorkspaceId === workspaceId) return;
		const ws = this.state.workspaces.find((w) => w.id === workspaceId);
		if (!ws) return;
		this.state = { ...this.state, activeWorkspaceId: workspaceId };
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(ws.root);	}

	renameTab(workspaceId: string, name: string): void {
		const trimmed = name.trim();
		if (!trimmed) return;
		this.state = {
			...this.state,
			workspaces: this.state.workspaces.map((w) =>
				w.id === workspaceId ? { ...w, name: trimmed } : w
			)
		};	}

	duplicateTab(workspaceId: string): void {
		const src = this.state.workspaces.find((w) => w.id === workspaceId);
		if (!src) return;
		const copy: Workspace = {
			id: makeWorkspace('').id,
			name: this._uniqueName(`${src.name} copy`),
			root: cloneWithNewIds(src.root)
		};
		const idx = this.state.workspaces.findIndex((w) => w.id === workspaceId);
		const workspaces = this.state.workspaces.slice();
		workspaces.splice(idx + 1, 0, copy);
		this.state = { workspaces, activeWorkspaceId: copy.id };
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(copy.root);	}

	closeTab(workspaceId: string): void {
		if (this.state.workspaces.length <= 1) return; // keep at least one tab
		const idx = this.state.workspaces.findIndex((w) => w.id === workspaceId);
		if (idx < 0) return;
		const workspaces = this.state.workspaces.filter((w) => w.id !== workspaceId);
		let activeWorkspaceId = this.state.activeWorkspaceId;
		if (activeWorkspaceId === workspaceId) {
			const neighbor = workspaces[Math.min(idx, workspaces.length - 1)];
			activeWorkspaceId = neighbor.id;
			this.maximizedPanelId = null;
			this.activePanelId = firstPanelId(neighbor.root);
		}
		this.state = { workspaces, activeWorkspaceId };	}

	/**
	 * Detach the dragged node (a panel or a whole tab) and return it plus the
	 * state with the source removed. Shared by every drop target — this is what
	 * makes a tab and a panel "the same thing": both yield a `LayoutNode` to
	 * re-home, with ids preserved so editor selections carry over. Returns null
	 * if the move would empty the last tab.
	 */
	private _takeNode(d: DragRef): { node: LayoutNode; state: WorkspaceState } | null {
		if (d.kind === 'tab') {
			const ws = this.state.workspaces.find((w) => w.id === d.workspaceId);
			if (!ws || this.state.workspaces.length <= 1) return null;
			const workspaces = this.state.workspaces.filter((w) => w.id !== d.workspaceId);
			const activeWorkspaceId =
				this.state.activeWorkspaceId === d.workspaceId
					? workspaces[0].id
					: this.state.activeWorkspaceId;
			return { node: ws.root, state: { workspaces, activeWorkspaceId } };
		}
		const ws = this.state.workspaces.find((w) => w.id === d.workspaceId);
		if (!ws) return null;
		const { root, removed } = extractPanel(ws.root, d.panelId);
		if (!removed) return null;
		if (root === null) {
			// the panel was the tab's only node → the tab goes with it
			if (this.state.workspaces.length <= 1) return null;
			const workspaces = this.state.workspaces.filter((w) => w.id !== d.workspaceId);
			const activeWorkspaceId =
				this.state.activeWorkspaceId === d.workspaceId
					? workspaces[0].id
					: this.state.activeWorkspaceId;
			return { node: removed, state: { workspaces, activeWorkspaceId } };
		}
		const workspaces = this.state.workspaces.map((w) =>
			w.id === d.workspaceId ? { ...w, root } : w
		);
		return { node: removed, state: { ...this.state, workspaces } };
	}

	/** Drop the dragged node (panel or tab) into the active layout by splitting
	 * `targetPanelId` along `direction`. Repositions a panel or merges a tab. */
	dropOnPanel(targetPanelId: string, direction: Direction, placeBefore: boolean): void {
		const d = this.dragging;
		this.dragging = null;
		if (!d) return;
		if (d.kind === 'panel' && d.panelId === targetPanelId) return; // onto itself
		const taken = this._takeNode(d);
		if (!taken) return;
		const { node, state } = taken;
		const active = state.workspaces.find((w) => w.id === state.activeWorkspaceId);
		if (!active) return;
		const root = insertNodeAtPanel(active.root, targetPanelId, direction, placeBefore, node);
		if (root === active.root) return; // target not in the active tree (e.g. self-drop)
		this.state = {
			...state,
			workspaces: state.workspaces.map((w) => (w.id === active.id ? { ...w, root } : w))
		};
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(node);
	}

	/** Drop the dragged panel onto the tab bar at `index` — it becomes a new
	 * tab. (Tabs dropped on the bar reorder instead — see reorderTab.) */
	dropPanelOnTabBar(index: number): void {
		const d = this.dragging;
		this.dragging = null;
		if (!d || d.kind !== 'panel') return;
		const taken = this._takeNode(d);
		if (!taken) return;
		const { node, state } = taken;
		const tab: Workspace = {
			id: makeWorkspace('').id,
			name: this._uniqueName('Layout'),
			root: node
		};
		const workspaces = state.workspaces.slice();
		workspaces.splice(Math.max(0, Math.min(index, workspaces.length)), 0, tab);
		this.state = { workspaces, activeWorkspaceId: tab.id };
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(node);
	}

	reorderTab(fromIndex: number, toIndex: number): void {
		const ws = this.state.workspaces.slice();
		if (fromIndex < 0 || fromIndex >= ws.length || toIndex < 0 || toIndex >= ws.length) return;
		const [moved] = ws.splice(fromIndex, 1);
		ws.splice(toIndex, 0, moved);
		this.state = { ...this.state, workspaces: ws };	}
}

let _store: WorkspaceStore | null = null;
export function workspace(): WorkspaceStore {
	if (!_store) _store = new WorkspaceStore();
	return _store;
}
