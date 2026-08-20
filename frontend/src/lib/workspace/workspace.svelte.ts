/**
 * Reactive workspace store — the browser's REPLICA of the manager's panel arrangement.
 *
 * The arrangement is the fifth CRDT doc root, held flat and id-keyed by the manager. This store
 * reads it (`syncFromDoc`) and rebuilds the tree the panel system draws (`arrangement.ts`); it
 * never edits an entry. Every gesture — split, close, resize, move, tab — goes out as a layout
 * COMMAND over `/control`, so the manager owns persistence, the broadcast to peers, and the undo
 * step, exactly as it does for the graph. There is no second write authority.
 *
 * What stays here is the VIEWPOINT: which page is in front, which panel is focused or maximized,
 * and each editor's sub-patch depth. Those belong to this client alone — pushing them into the doc
 * would drag a peer's phone out of the sub-patch it is three levels into, and would dirty a patch
 * for looking around. They persist through `set_viewpoint`, which the manager stores and rides into
 * the `.gfi` without ever converging or dirtying it. Persistence and dirtiness are separate axes.
 */
import {
	collectPanels,
	DEFAULT_PANEL_TYPE,
	findNode,
	findPanel,
	firstPanelId,
	firstPanelIn,
	resizeFractions,
	type Direction,
	type LayoutNode,
	type Workspace,
	type WorkspaceState
} from './model';
import { asStateObject } from './panelState';
import type { LayoutHost, TabRef } from './host';

/** A drag in progress. A panel and a tab are both just a subtree being moved — the only difference
 * is which id names it, which `_subtreeOf` knows how to answer. */
export type DragRef =
	| { kind: 'panel'; workspaceId: string; panelId: string }
	| { kind: 'tab'; workspaceId: string };

/**
 * Why a panel write happened — the axis that decides whether the patch now differs from the file on
 * disk. Since the cutover it is not a classification the manager has to be told: it is WHICH OP the
 * write becomes, so the taxonomy holds by construction.
 *
 * - `'authored'` — the user edited the arrangement (a viewer kind, a bound slot). It becomes a
 *   `set_panel` command: undoable, converged to every peer, and it dirties.
 * - `'navigation'` — the user only changed what they are LOOKING at (entering a sub-patch). It
 *   becomes viewpoint: stored for this client, never converged, never dirtying.
 *
 * The routing follows the WRITE, never the device — phone and desktop share this one rule.
 */
export type LayoutIntent = 'navigation' | 'authored';

/** What `set_viewpoint` stores for this client, and what a reload gets back. */
export interface Viewpoint {
	tab?: string;
	panel?: string;
	paths?: Record<string, string>;
}

/** A split's children's shares, wherever it is in the strip — the baseline a resize drag adjusts. */
function fractionsOf(tabs: Workspace[], split: string): number[] {
	for (const t of tabs) {
		const n = findNode(t.root, split);
		if (n?.kind === 'split') return n.sizes;
	}
	return [];
}

/** What the panel system draws before the replica has pulled from the manager. The ids are the
 * manager's own first-mint spelling, so the pre-sync frame and the synced one draw the same thing
 * and nothing re-keys under the user. */
const UNSYNCED: Workspace[] = [
	{
		id: 'tab-1',
		name: 'Tab 1',
		root: { kind: 'panel', id: 'panel-2', panelType: DEFAULT_PANEL_TYPE, state: undefined }
	}
];

/** What the panel system talks to before a consumer has installed a host: everything draws, every
 * gesture is refused. A `null` host would make each call site a guard instead. */
const REFUSING: LayoutHost = {
	addTab: async () => null,
	removeTab: async () => false,
	renameTab: async () => false,
	reorderTab: async () => false,
	splitPanel: async () => null,
	removePanel: async () => false,
	resizeSplit: async () => false,
	setPanel: async () => false,
	movePanel: async () => false
};

class WorkspaceStore {
	/** The manager's arrangement, mirrored from the doc root. Read-only: a write is a command. */
	private _tabs = $state<Workspace[]>([]);
	/** The shares a splitter drag is currently drawing, before it commits. A resize is one
	 * continuous gesture, so the override lives here for its duration and lands as ONE
	 * `resize_split` on pointer-up — never a command per pointermove. */
	private _drag = $state<{ split: string; sizes: number[] } | null>(null);
	/** The shares a commit put on the wire, held until the delta answering it lands. It keeps the
	 * drawn shares from snapping back in the frame between the reply and the doc arriving (the reply
	 * is sent first), and it is what the NEXT commit's "nothing changed" check compares against — the
	 * replica is still a commit behind, so a drag returning the split to its pre-commit shares would
	 * otherwise read as a no-op and be dropped. */
	private _sent: { split: string; sizes: number[] } | null = null;
	/** Whether the pointer is still on the seam. A delta landing mid-gesture — the previous commit's
	 * own, or a peer's — must not retire the override the finger is drawing with. */
	private _dragLive = false;
	/** Viewpoint: the page in front. Null falls back to the first, which is what a fresh client and
	 * a page a peer closed both want. */
	private _page = $state<string | null>(null);
	/** Viewpoint: per panel id, the sub-patch path that editor is inside. Held OUT of the panel's
	 * shared state bag — that separation is what keeps peer isolation and navigation-must-not-dirty
	 * true by construction rather than by classification. */
	private _paths = $state<Record<string, string>>({});
	/** The page and root panel a just-accepted `add_tab` minted, brought forward once the
	 * doc catches up. The manager answers with the ids before the CRDT delta carrying them arrives,
	 * so they are known first but cannot be drawn yet. */
	private _wantTab: TabRef | null = null;
	/** Last-focused panel id — keyboard shortcuts scope to this. */
	activePanelId = $state<string | null>(null);
	/** Viewpoint: the maximized panel, PER PAGE. A page keeps its own — maximizing on one tab and
	 * looking at another must not undo it, and coming back must find it as it was left. It was one
	 * scalar for the whole client, which made the two pages one state and made every tab switch a
	 * reset. Session-scoped on purpose: it is deliberately not in `viewpoint()`, so it reaches
	 * neither a peer nor the `.gfi`. */
	private _max = $state<Record<string, string>>({});
	/** The panel or tab currently being dragged. While set, panels show edge drop zones and the tab
	 * bar accepts the drop. */
	dragging = $state<DragRef | null>(null);
	/** Bumped whenever the viewpoint changes, so the shell can persist it debounced. */
	viewpointEpoch = $state(0);
	/** Whoever owns the arrangement. Every gesture below is raised through it and nothing here
	 * writes a tree — see `./host`. Until one is installed the panel system draws and refuses,
	 * which is what a consumer that has not wired itself up should see. */
	private _host: LayoutHost = REFUSING;

	configureHost(host: LayoutHost): void {
		this._host = host;
	}

	/** The tree as DRAWN: the manager's, with this client's two overlays — the in-flight resize a
	 * finger is still describing, and each editor's sub-patch depth. Both are viewpoint, so neither
	 * is in the manager's copy and neither survives a rebuild from it. */
	private _workspaces = $derived.by(() => {
		const paths = this._paths;
		const drag = this._drag;
		const overlay = (n: LayoutNode): LayoutNode => {
			if (n.kind === 'panel') {
				const path = paths[n.id];
				return path === undefined
					? n
					: { ...n, state: { ...asStateObject(n.state), subpatchPath: path } };
			}
			const sizes = drag?.split === n.id && drag.sizes.length === n.children.length ? drag.sizes : n.sizes;
			return { ...n, sizes, children: n.children.map(overlay) };
		};
		const tabs = this._tabs.map((w) => ({ ...w, root: overlay(w.root) }));
		return tabs.length > 0 ? tabs : UNSYNCED;
	});

	/** The tree the panel system renders. Derived, not held — the manager's copy is the state. */
	state = $derived<WorkspaceState>({
		workspaces: this._workspaces,
		activeWorkspaceId: this.active.id
	});

	/** Whether a panel can be torn off into a tab of its own. False when the host cannot express it
	 * — a standalone `<Panels>` has no strip to tear onto — and the strip reads this to decide
	 * whether to OFFER the drop at all, so the gesture is absent rather than refused. */
	get canTearOff(): boolean {
		return this._host.tabFromPanel !== undefined;
	}

	/** The maximized panel on the page in front, or null when that page is showing its layout. */
	get maximizedPanelId(): string | null {
		const page = this.active?.id;
		return (page ? this._max[page] : undefined) ?? null;
	}

	get active(): Workspace {
		const all = this._workspaces;
		return all.find((w) => w.id === this._page) ?? all[0];
	}

	// --- the replica ---------------------------------------------------------

	/** Adopt the arrangement the manager mirrored, and prune any viewpoint it invalidated — a panel
	 * WE focused that a peer just closed, a page that went with it. This is also where an in-flight
	 * resize override retires: the split's own shares moved, so the drawn tree is the manager's
	 * again. */
	syncFromDoc(tabs: Workspace[]): void {
		const prev = this._tabs;
		this._tabs = tabs;

		const s = this._sent;
		if (s) {
			const before = fractionsOf(prev, s.split);
			const after = fractionsOf(tabs, s.split);
			if (before.length !== after.length || before.some((v, i) => v !== after[i])) {
				this._sent = null;
				if (!this._dragLive) this._drag = null;
			}
		}
		const want = this._wantTab;
		if (want && tabs.some((t) => t.id === want.tab)) {
			this._wantTab = null;
			this._page = want.tab;
			this._focus(want.panel);
		}
		// An EMPTY arrangement is a generation boundary, never a settled tree — the reset that hands
		// one over is followed by the manager's real document, and a replica before its first pull
		// holds one too. Pruning against it invalidates every id there is, the viewpoint the
		// snapshot just restored included.
		if (tabs.length === 0) return;
		const live = (id: string): boolean => tabs.some((t) => t.id === id || !!findNode(t.root, id));
		if (this._page !== null && !live(this._page)) this._page = null;
		const root = this.active?.root;
		if (!root) return;
		if (!this.activePanelId || !findPanel(root, this.activePanelId)) {
			this.activePanelId = firstPanelId(root);
		}
		for (const [tab, panel] of Object.entries(this._max)) {
			if (!live(tab) || !live(panel)) delete this._max[tab];
		}
		for (const id of Object.keys(this._paths)) {
			if (!live(id)) delete this._paths[id];
		}
	}

	/** Restore the viewpoint this client last stored (it rides the `.gfi` and the snapshot, but is
	 * never converged). Ids that no longer exist are simply not adopted. */
	restoreViewpoint(vp: unknown): void {
		const v = vp as Viewpoint | null;
		if (!v || typeof v !== 'object') return;
		if (typeof v.tab === 'string') this._page = v.tab;
		if (typeof v.panel === 'string') this.activePanelId = v.panel;
		if (v.paths && typeof v.paths === 'object') this._paths = { ...v.paths };
	}

	/** What `set_viewpoint` stores. Plain JSON: the shell pushes it debounced. */
	viewpoint(): Viewpoint {
		return {
			tab: this.active?.id,
			panel: this.activePanelId ?? undefined,
			paths: $state.snapshot(this._paths)
		};
	}

	private _viewpointChanged(): void {
		this.viewpointEpoch += 1;
	}

	// --- gestures ------------------------------------------------------------

	/** After a structural layout change, drop the maximized view and focus `panelId`. */
	private _focus(panelId: string): void {
		this.activePanelId = panelId;
		this._viewpointChanged();
	}

	/** The same, for a page arriving whole: its first panel (firstPanelId returns '' when empty). */
	private _focusFirst(root: LayoutNode): void {
		this._focus(firstPanelId(root));
	}

	// --- layout mutations ----------------------------------------------------

	/** Split a panel. The new panel is `empty` — the user picks its content from the empty panel's
	 * buttons rather than inheriting the source's type. `fraction` is the new panel's share. */
	split(panelId: string, direction: Direction, placeBefore = false, fraction = 0.5): void {
		void this._host.splitPanel(panelId, direction, placeBefore, fraction).then((fresh) => {
			if (fresh === null) return;
			this.activePanelId = fresh;
			this._viewpointChanged();
		});
	}

	close(panelId: string): void {
		void this._host.removePanel(panelId);
	}

	/** A splitter drag fires this per pointermove. It draws locally — `containerPx` is the split's
	 * measured size along its axis, the denominator of the pixel floor (D-R10) — and nothing leaves
	 * the client until `commitResize`. */
	resize(splitId: string, dividerIndex: number, delta: number, containerPx = 0): void {
		const base =
			this._drag?.split === splitId ? this._drag.sizes : fractionsOf(this._tabs, splitId);
		if (base.length === 0) return;
		this._dragLive = true;
		this._drag = { split: splitId, sizes: resizeFractions(base, dividerIndex, delta, containerPx) };
	}

	/** Pointer-up: the shares the drag drew become ONE command, and therefore one ctrl-Z. */
	commitResize(splitId: string): void {
		const d = this._drag;
		// The pointer is up either way. `_sent` outlives the drop: it is still the last thing this
		// client put on the wire for that split, and therefore still the honest baseline.
		this._dragLive = false;
		const drop = (): void => {
			this._drag = null;
		};
		if (!d || d.split !== splitId) {
			drop();
			return;
		}
		// What was last SENT for this split, falling back to the replica. Comparing against the
		// replica alone would drop a second drag that returns the split to its pre-commit shares,
		// because the replica is still showing exactly those.
		const before =
			this._sent?.split === splitId ? this._sent.sizes : fractionsOf(this._tabs, splitId);
		const same = before.length === d.sizes.length && before.every((s, i) => s === d.sizes[i]);
		if (same) {
			drop();
			return;
		}
		this._sent = { split: splitId, sizes: d.sizes };
		void this._host.resizeSplit(splitId, d.sizes).then((ok) => {
			// A refusal never landed, so it is not a baseline either.
			if (!ok) {
				this._sent = null;
				drop();
			}
		});
	}

	setType(panelId: string, panelType: string): void {
		void this._host.setPanel(panelId, { type: panelType });
	}

	/**
	 * Write a panel's opaque state. `intent` routes it, and that routing IS the dirty taxonomy:
	 * `'navigation'` (the sub-patch path) stays viewpoint and never leaves as a layout op, while an
	 * authored write becomes `set_panel` — one command, one undo step, converged to peers.
	 * `label` names that step so the undo button reads like the click.
	 */
	setPanelState(
		panelId: string,
		state: unknown,
		intent: LayoutIntent = 'authored',
		label = 'Change panel'
	): void {
		const bag = asStateObject(state);
		if (intent === 'navigation') {
			const path = bag.subpatchPath;
			if (typeof path === 'string') this._paths[panelId] = path;
			else delete this._paths[panelId];
			this._viewpointChanged();
			return;
		}
		// The sub-patch path is viewpoint and must not ride a shared write.
		const { subpatchPath: _drop, ...shared } = bag;
		void this._host.setPanel(panelId, { state: shared }, label);
	}

	setActive(panelId: string): void {
		if (this.activePanelId === panelId) return;
		this.activePanelId = panelId;
		this._viewpointChanged();
	}

	/* The three panel-state edits below each name ONLY the key they change: `set_panel` merges,
	 * so reading the bag back first would buy nothing and cost the class the merge exists to kill —
	 * a second write inside the first's round trip replacing a bag it never saw the first land in.
	 * Addressing is `setPanelState`'s, which resolves the page from the whole arrangement, so a panel
	 * on a page in the background is written just like one in front. */

	/** Bind a node to a linkable panel — a node dragged onto it, or one picked from the bar's
	 * dropdown (`panels/NodeSelect`). Both doors land here, so they cannot behave differently.
	 *
	 * The slot goes with it: a Viewer/Metadata panel reads `state.slot` off the node it is bound to,
	 * and the two names have nothing to do with each other across a rebind. Clearing it in the SAME
	 * merged write settles the panel in one op (one undo step) on the slot each panel already falls
	 * back to — its node's first output — instead of persisting a slot name the new node has never
	 * had. */
	linkNodeToPanel(panelId: string, nodeUid: string): void {
		this.setPanelState(panelId, { node: nodeUid, slot: null }, 'authored', 'Bind node to panel');
	}

	/** Release a linkable panel's bound node — the ✕ in NodeLinkedPanel's bar and ConsolePanel's
	 * filter chip. An explicit null is how a merged write clears a key. */
	unlinkNodeFromPanel(panelId: string): void {
		this.setPanelState(panelId, { node: null }, 'authored', 'Unbind node from panel');
	}

	/** Pick the output slot a Viewer / Metadata panel reads from its bound node. */
	setPanelSlot(panelId: string, slot: string): void {
		this.setPanelState(panelId, { slot }, 'authored', 'Select slot');
	}

	toggleMaximize(panelId: string): void {
		const tab = this._tabs.find((t) => findNode(t.root, panelId))?.id;
		if (!tab) return;
		if (this._max[tab] === panelId) delete this._max[tab];
		else this._max[tab] = panelId;
		this._viewpointChanged();
	}

	/** End the maximize on the page in front, so a panel this client is about to show is visible.
	 * The one caller is the shell answering an agent's close (`editor/TopBar`), which has to bring a
	 * specific panel to the front and cannot do that under a maximized neighbour. */
	exitMaximize(): void {
		const page = this.active?.id;
		if (page && this._max[page] !== undefined) {
			delete this._max[page];
			this._viewpointChanged();
		}
	}

	// --- tabs --------------------------------------------------------------

	/** Add a tab. `panelType` is the agent façade's door onto "a tab showing X"; the strip's own ＋
	 * births whatever the host's default is. The NAME is not asked for — a label is not addressing,
	 * and only the host can see what the other tabs are called. */
	addTab(panelType?: string): void {
		void this._host.addTab({ panelType }).then((born) => {
			if (born) this._wantTab = born;
		});
	}

	/** D-R11: switching layout tabs is NAVIGATION. It changes which arrangement is in front, not
	 * what any panel holds — the same "looking elsewhere" as entering a sub-patch. Creating,
	 * renaming, reordering or closing a tab is still authoring; only the selection is a look. */
	selectTab(workspaceId: string): void {
		if (this._page === workspaceId) return;
		const ws = this._workspaces.find((w) => w.id === workspaceId);
		if (!ws) return;
		this._page = workspaceId;
		this._focusFirst(ws.root);
	}

	renameTab(workspaceId: string, name: string): void {
		const trimmed = name.trim();
		const from = this._tabs.find((t) => t.id === workspaceId)?.name;
		if (!trimmed || !from || trimmed === from) return;
		void this._host.renameTab(workspaceId, trimmed);
	}

	closeTab(workspaceId: string): void {
		if (!this._tabs.some((t) => t.id === workspaceId)) return;
		// Closing the tab in front moves us to its NEIGHBOUR, not to the strip's first — the frozen
		// behaviour, and viewpoint, so it lands now rather than waiting for the delta. Without it the
		// fallback (`?? all[0]`) silently rewrote the gesture.
		if (this._page === workspaceId || this._page === null) {
			const rest = this._workspaces.filter((w) => w.id !== workspaceId);
			const idx = this._workspaces.findIndex((w) => w.id === workspaceId);
			const neighbor = rest[Math.min(idx, rest.length - 1)];
			if (neighbor) {
				this._page = neighbor.id;
				this._focusFirst(neighbor.root);
			}
		}
		void this._host.removeTab(workspaceId);
	}

	reorderTab(fromIndex: number, toIndex: number): void {
		const tab = this._workspaces[fromIndex]?.id;
		if (tab === undefined || toIndex < 0 || toIndex >= this._workspaces.length) return;
		void this._host.reorderTab(tab, toIndex);
	}

	/** The id of the subtree a drag names: a panel is a subtree of one, a tab drag carries the
	 * page's whole root. Null when the drag's source has gone. */
	private _subtreeOf(d: DragRef): string | null {
		if (d.kind === 'panel') {
			return this._tabs.some((t) => findNode(t.root, d.panelId)) ? d.panelId : null;
		}
		return this._tabs.find((t) => t.id === d.workspaceId)?.root.id ?? null;
	}

	/** Drop the dragged node (panel or tab) into the layout by splitting `targetPanelId` along
	 * `direction`. Repositions a panel or merges a tab — ONE op, so it is one undo step, and taking
	 * a page's last panel takes the page with it. */
	dropOnPanel(targetPanelId: string, direction: Direction, placeBefore: boolean): void {
		const d = this.dragging;
		this.dragging = null;
		if (!d) return;
		if (d.kind === 'panel' && d.panelId === targetPanelId) return; // onto itself
		const subtree = this._subtreeOf(d);
		if (!subtree || subtree === targetPanelId) return;
		// A drop lands on a tab already in front, so only the FOCUS moves — onto the panel the user
		// just carried there, which is the one they are now working in.
		const node = this._tabs.map((t) => findNode(t.root, subtree)).find(Boolean);
		const focus = node ? firstPanelIn(node) : null;
		void this._host.movePanel(subtree, targetPanelId, direction, placeBefore, 0.5).then((ok) => {
			if (ok && focus !== null) this._focus(focus);
		});
	}

	/** Drop the dragged panel onto the tab bar at `index` — it becomes a new tab, built AROUND the
	 * panel it carries. (Tabs dropped on the bar reorder instead — see reorderTab.) */
	dropPanelOnTabBar(index: number): void {
		const d = this.dragging;
		this.dragging = null;
		if (!d || d.kind !== 'panel') return;
		const subtree = this._subtreeOf(d);
		// The one gesture that spans both halves, so it exists only where they are composed. A host
		// that cannot express it does not offer the drag rather than failing it.
		if (!subtree || !this._host.tabFromPanel) return;
		void this._host.tabFromPanel(subtree, index).then((born) => {
			if (born) this._wantTab = born;
		});
	}

	/** Every panel currently bound to `uid`, for the agent façade and the e2e. */
	panelsBoundTo(uid: string): string[] {
		const out: string[] = [];
		for (const w of this._workspaces) {
			for (const p of collectPanels(w.root)) {
				if (asStateObject(p.state).node === uid) out.push(p.id);
			}
		}
		return out;
	}
}

let _store: WorkspaceStore | null = null;
export function workspace(): WorkspaceStore {
	if (!_store) _store = new WorkspaceStore();
	return _store;
}
