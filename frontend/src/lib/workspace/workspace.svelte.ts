/**
 * Reactive workspace store — the only stateful layer of the panel system.
 *
 * Holds the `WorkspaceState` (tabs + their layout trees) plus ephemeral UI
 * state (which panel is active, which is maximized). All structural changes go
 * through the pure ops in `model.ts`; this layer just owns the `$state` and
 * picks sensible follow-up selection.
 *
 * The layout is *not* persisted in the browser — no localStorage, no session
 * storage. It lives only in memory here, is pushed to the manager as it
 * changes (AppShell → `set_layout`), and is embedded in the `.gfi` on save. A
 * fresh manager session therefore starts at the default layout; `GraphStore`
 * detects a backend restart (via the snapshot `instance_id`) and calls
 * `reset()` so a stale layout never lingers across a kill/restart.
 */
import {
	clearNodeRef,
	closePanel,
	cloneWithNewIds,
	collectPanels,
	defaultWorkspaceState,
	DEFAULT_PANEL_TYPE,
	EMPTY_PANEL_TYPE,
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
import { linkedNodeName, withLinkedNode } from './panelState';
import { history, type LayoutActionKind } from '$lib/stores/history.svelte';
import { captureNavContext } from './navContext';

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

	constructor() {
		this.activePanelId = firstPanelId(this.active.root);
	}

	/** After a structural layout change, drop the maximized view and focus the
	 * first panel of `root` (firstPanelId returns '' when empty). */
	private _focusFirst(root: LayoutNode): void {
		this.maximizedPanelId = null;
		this.activePanelId = firstPanelId(root);
	}

	/** Drop all layout state back to a single default panel. Called when a
	 * fresh backend session connects without a layout of its own, so the panel
	 * arrangement from the previous session doesn't linger in the open tab. */
	reset(): void {
		this.state = defaultWorkspaceState();
		this._focusFirst(this.active.root);
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

	/** Restore an exact `WorkspaceState` snapshot for undo/redo. Unlike
	 * `hydrate`, this does NOT reseed ids or migrate — the snapshot is replayed
	 * verbatim so panel/tab ids (and the selections keyed to them) are preserved.
	 * Ephemeral view state (maximize) is dropped; focus is reset to a sensible
	 * default and then overridden by the action's NavContext. */
	restore(state: WorkspaceState): void {
		this.state = state;
		this._focusFirst(this.active.root);
	}

	/** Apply a layout restored from a `.gfi` patch (or any external source). */
	hydrate(state: unknown): void {
		if (!isValidState(state)) return;
		reseedIds(state);
		// Back-compat: the old "errors" panel is now the generalized "console".
		const migrate = (node: LayoutNode): void => {
			if (node.kind === 'split') node.children.forEach(migrate);
			else if (node.panelType === 'errors') node.panelType = 'console';
		};
		for (const ws of state.workspaces) migrate(ws.root);
		this.state = state;
		this._focusFirst(this.active.root);
	}

	/** Run a tracked layout mutation `fn`, recording one history entry iff it
	 * actually changed the state tree (a no-op split / blocked close records
	 * nothing). `coalesceKey` merges a continuous gesture into one entry. */
	private _tracked(kind: LayoutActionKind, label: string, fn: () => void, coalesceKey?: string): void {
		const before = this.serialize();
		const prev = this.state;
		fn();
		if (this.state !== prev && !history().isSuspended) {
			history().record({
				kind,
				domain: 'layout',
				label,
				context: captureNavContext(),
				coalesceKey,
				payload: { before, after: this.serialize() }
			});
		}
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

	/** Split a panel. The new panel is `empty` by default — the user picks its
	 * content from the empty panel's buttons rather than inheriting the source's
	 * type. `fraction` is the new panel's share of the split (0.5 = even). */
	split(
		panelId: string,
		direction: Direction,
		placeBefore = false,
		fraction = 0.5,
		newType?: string
	): void {
		this._tracked('split_panel', 'Split panel', () => {
			const ws = this.active;
			const type = newType ?? EMPTY_PANEL_TYPE;
			const { root, newPanelId } = splitPanel(ws.root, panelId, direction, placeBefore, type, fraction);
			if (root === ws.root) return;
			this._setRoot(ws.id, root);
			if (newPanelId) this.activePanelId = newPanelId;
		});
	}

	close(panelId: string): void {
		this._tracked('close_panel', 'Close panel', () => {
			const ws = this.active;
			const root = closePanel(ws.root, panelId);
			if (!root) return;
			this._setRoot(ws.id, root);
			if (this.maximizedPanelId === panelId) this.maximizedPanelId = null;
			if (this.activePanelId === panelId) this.activePanelId = firstPanelId(root);
		});
	}

	resize(splitId: string, dividerIndex: number, delta: number): void {
		// A splitter drag fires this per mousemove — coalesce the whole drag into
		// one undo step via a stable per-divider key.
		this._tracked(
			'resize_split',
			'Resize',
			() => this._updateActiveRoot((root) => resizeSplit(root, splitId, dividerIndex, delta)),
			`resize:${splitId}:${dividerIndex}`
		);
	}

	setType(panelId: string, panelType: string): void {
		this._tracked('set_panel_type', 'Change panel', () => {
			const ds = getPanelType(panelType)?.defaultState?.();
			this._updateActiveRoot((root) => setPanelType(root, panelId, panelType, ds));
		});
	}

	setPanelState(panelId: string, state: unknown): void {
		this._updateActiveRoot((root) => setPanelState(root, panelId, state));
	}

	setActive(panelId: string): void {
		if (this.activePanelId !== panelId) this.activePanelId = panelId;
	}

	/** Bind a node to a linkable panel (merges `node` into its state, keeping
	 * any slot/kind). Called when a node is dragged onto the panel. */
	linkNodeToPanel(panelId: string, nodeName: string): void {
		this._tracked('link_node_to_panel', 'Bind node to panel', () => {
			this._updateActiveRoot((root) => {
				const p = findPanel(root, panelId);
				if (!p) return root;
				return setPanelState(root, panelId, withLinkedNode(p.state, nodeName));
			});
		});
	}

	/** Every panel currently bound to `nodeName`, with a snapshot of its state.
	 * The undo system captures these before a node delete so undoing the delete
	 * can re-bind the panels that `clearNodeRefs` emptied. */
	panelsBoundTo(nodeName: string): Array<{ panelId: string; state: unknown }> {
		const out: Array<{ panelId: string; state: unknown }> = [];
		for (const w of this.state.workspaces) {
			for (const p of collectPanels(w.root)) {
				if (linkedNodeName(p.state) === nodeName) {
					out.push({ panelId: p.id, state: $state.snapshot(p.state) });
				}
			}
		}
		return out;
	}

	/** Unlink a deleted node from every panel bound to it (empties them). */
	clearNodeRefs(nodeName: string): void {
		let changed = false;
		const workspaces = this.state.workspaces.map((w) => {
			const root = clearNodeRef(w.root, nodeName);
			if (root === w.root) return w;
			changed = true;
			return { ...w, root };
		});
		if (changed) this.state = { ...this.state, workspaces };
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
		this._tracked('add_tab', 'Add tab', () => {
			const ws = makeWorkspace(this._uniqueName('Layout'), panelType);
			this.state = {
				workspaces: [...this.state.workspaces, ws],
				activeWorkspaceId: ws.id
			};
			this._focusFirst(ws.root);
		});
	}

	selectTab(workspaceId: string): void {
		if (this.state.activeWorkspaceId === workspaceId) return;
		const ws = this.state.workspaces.find((w) => w.id === workspaceId);
		if (!ws) return;
		this.state = { ...this.state, activeWorkspaceId: workspaceId };
		this._focusFirst(ws.root);
	}

	renameTab(workspaceId: string, name: string): void {
		this._tracked('rename_tab', 'Rename tab', () => {
			const trimmed = name.trim();
			if (!trimmed) return;
			this.state = {
				...this.state,
				workspaces: this.state.workspaces.map((w) =>
					w.id === workspaceId ? { ...w, name: trimmed } : w
				)
			};
		});
	}

	duplicateTab(workspaceId: string): void {
		this._tracked('duplicate_tab', 'Duplicate tab', () => {
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
			this._focusFirst(copy.root);
		});
	}

	closeTab(workspaceId: string): void {
		this._tracked('close_tab', 'Close tab', () => {
			if (this.state.workspaces.length <= 1) return; // keep at least one tab
			const idx = this.state.workspaces.findIndex((w) => w.id === workspaceId);
			if (idx < 0) return;
			const workspaces = this.state.workspaces.filter((w) => w.id !== workspaceId);
			let activeWorkspaceId = this.state.activeWorkspaceId;
			if (activeWorkspaceId === workspaceId) {
				const neighbor = workspaces[Math.min(idx, workspaces.length - 1)];
				activeWorkspaceId = neighbor.id;
				this._focusFirst(neighbor.root);
			}
			this.state = { workspaces, activeWorkspaceId };
		});
	}

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
			if (!ws) return null;
			const state = this._removeWorkspace(d.workspaceId);
			if (!state) return null;
			return { node: ws.root, state };
		}
		const ws = this.state.workspaces.find((w) => w.id === d.workspaceId);
		if (!ws) return null;
		const { root, removed } = extractPanel(ws.root, d.panelId);
		if (!removed) return null;
		if (root === null) {
			// the panel was the tab's only node → the tab goes with it
			const state = this._removeWorkspace(d.workspaceId);
			if (!state) return null;
			return { node: removed, state };
		}
		const workspaces = this.state.workspaces.map((w) =>
			w.id === d.workspaceId ? { ...w, root } : w
		);
		return { node: removed, state: { ...this.state, workspaces } };
	}

	/** Remove an entire workspace tab, choosing a fallback active tab. Returns null
	 * when it's the last tab (which must be kept). The shared body of `_takeNode`'s
	 * tab-drag and last-panel-removal branches. */
	private _removeWorkspace(id: string): WorkspaceState | null {
		if (this.state.workspaces.length <= 1) return null;
		const workspaces = this.state.workspaces.filter((w) => w.id !== id);
		const activeWorkspaceId =
			this.state.activeWorkspaceId === id ? workspaces[0].id : this.state.activeWorkspaceId;
		return { workspaces, activeWorkspaceId };
	}

	/** Drop the dragged node (panel or tab) into the active layout by splitting
	 * `targetPanelId` along `direction`. Repositions a panel or merges a tab. */
	dropOnPanel(targetPanelId: string, direction: Direction, placeBefore: boolean): void {
		this._tracked('move_panel', 'Move panel', () => {
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
			this._focusFirst(node);
		});
	}

	/** Drop the dragged panel onto the tab bar at `index` — it becomes a new
	 * tab. (Tabs dropped on the bar reorder instead — see reorderTab.) */
	dropPanelOnTabBar(index: number): void {
		this._tracked('move_panel', 'Move panel to new tab', () => {
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
			this._focusFirst(node);
		});
	}

	reorderTab(fromIndex: number, toIndex: number): void {
		this._tracked('reorder_tab', 'Reorder tabs', () => {
			const ws = this.state.workspaces.slice();
			if (fromIndex < 0 || fromIndex >= ws.length || toIndex < 0 || toIndex >= ws.length) return;
			const [moved] = ws.splice(fromIndex, 1);
			ws.splice(toIndex, 0, moved);
			this.state = { ...this.state, workspaces: ws };
		});
	}
}

let _store: WorkspaceStore | null = null;
export function workspace(): WorkspaceStore {
	if (!_store) _store = new WorkspaceStore();
	return _store;
}
