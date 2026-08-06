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
	uid,
	type Direction,
	type LayoutNode,
	type Workspace,
	type WorkspaceState
} from './model';
import { asStateObject, linkedNodeName, withLinkedNode } from './panelState';
import { history, type LayoutActionKind } from '$lib/stores/history.svelte';
import { captureNavContext } from './navContext';

/** A drag in progress. A panel and a tab are both just a `LayoutNode` being
 * moved — the only difference is where it came from, which `_takeNode` knows
 * how to detach. */
export type DragRef =
	| { kind: 'panel'; workspaceId: string; panelId: string }
	| { kind: 'tab'; workspaceId: string };

/**
 * Why a layout write happened — the axis that decides whether the patch now differs from the file
 * on disk. Persistence is a SEPARATE axis: every layout write is pushed to the manager and rides
 * the `.gfi` either way.
 *
 * - `'authored'` — the user edited the arrangement: split/close/resize a panel, add or rename a
 *   tab, pick a viewer kind or an output slot, unlink a node. The patch really did change.
 * - `'navigation'` — the user only changed what they are LOOKING at (entering a sub-patch,
 *   switching layout tabs, an undo/redo re-orientation), or the manager echoed its own layout back
 *   at us on hello/load. Persisted, but the patch still matches disk, so it must not raise the
 *   unsaved dot or the unload guard.
 *
 * The classification follows the WRITE, never the device — phone and desktop share this one rule.
 */
export type LayoutIntent = 'navigation' | 'authored';

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
	/** Folded intent of the layout writes not yet pushed to the manager. Plain (not `$state`) —
	 * it is read once at push time and never rendered. */
	private _pendingIntent: LayoutIntent = 'navigation';
	/** Set when `this.state` was last replaced by a PEER's arrangement rather than by a local
	 * write. Every state replacement re-triggers AppShell's debounced push, and pushing a peer's
	 * layout back is both a pointless round trip and a way to overwrite the manager's copy with
	 * THIS client's navigation fields — so the shell reads this at push time and drops that one
	 * push. Any local write clears it (see `_mark`/`_replaced`), so a real edit landing inside the
	 * same debounce window still reaches the manager. Plain (not `$state`) for the same reason as
	 * `_pendingIntent`. */
	private _remoteApplied = false;

	constructor() {
		this.activePanelId = firstPanelId(this.active.root);
	}

	/** Classify the layout write that just happened. Authoring wins a mixed debounce window: if
	 * anything in it was an edit, the patch really did change — so this can only RAISE. The two
	 * wholesale replacements below assign instead (see `_replaced`). */
	private _mark(intent: LayoutIntent): void {
		// Any LOCAL write is this client's own and has to reach the manager — navigation included,
		// since persistence is the other axis. So it always clears the remote-apply latch.
		this._remoteApplied = false;
		if (intent === 'authored') this._pendingIntent = 'authored';
	}

	/** `hydrate`/`reset` replace `this.state` outright rather than editing it, so they discard any
	 * pending edit ALONG WITH the state that carried it — the fold has nothing left to be about.
	 * Marking (which can only raise) let that edit ride the next push and dirty a patch that
	 * matches disk: on `load` the CRDT delta can beat the queued JSON events, and `clearNodeRefs`
	 * then marks authored moments before `graph_replaced` hydrates. A write that lands the other
	 * way round — after the replacement — edited the state that is now live and still counts. */
	private _replaced(): void {
		this._pendingIntent = 'navigation';
		// A load / reset is still a local event the manager has to hear about, so the latch goes
		// too. `applyRemoteLayout` re-arms it AFTER calling this.
		this._remoteApplied = false;
	}

	/** A whole patch was loaded — a new backend session, or the Load button. The graph the pending
	 * fold was about is gone, so whatever it recorded cannot describe a difference from the file
	 * that just arrived.
	 *
	 * `hydrate`/`reset` already drain the fold, and cover the load whose layout applies. This is
	 * for the loads where NEITHER runs: a layout-less `.gfi` (the engine supports one) landing on
	 * a same-session `graph_replaced` takes no branch, and a malformed layout makes `hydrate`
	 * refuse before it drains. Either way the load's own CRDT delta has already emptied every
	 * panel bound to a vanished uid — `clearNodeRefs`, marking authored, correctly, for a node
	 * DELETE that in this case is just the load. Called only from the graph store's wholesale-load
	 * path, which a transient same-session reconnect deliberately does not reach: an edit made
	 * while the socket was down is still an edit. */
	patchLoaded(): void {
		this._replaced();
	}

	/** The folded intent of every layout write since the last call, then reset. AppShell takes
	 * this when it pushes `set_layout`, so the manager can tell an edit from a look. */
	takeLayoutIntent(): LayoutIntent {
		const intent = this._pendingIntent;
		this._pendingIntent = 'navigation';
		return intent;
	}

	/** Whether the pending push exists ONLY because a peer's arrangement was applied, then reset.
	 * AppShell asks this before pushing: true means drop the push (see `_remoteApplied`). */
	takeRemoteApplied(): boolean {
		const remote = this._remoteApplied;
		this._remoteApplied = false;
		return remote;
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
		this._replaced(); // a blank session's default arrangement is nobody's edit
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
		this._mark('authored'); // undoing/redoing an edit to the arrangement is still an edit
		this._focusFirst(this.active.root);
	}

	/** Ready a layout blob minted OUTSIDE this client — a `.gfi`, or a peer's push. Advances the id
	 * counter past every id it already uses so a panel we mint next cannot collide, and migrates the
	 * legacy "errors" panel onto the generalized "console". Shared by `hydrate` and
	 * `applyRemoteLayout` so the two cannot drift. */
	private _adopt(state: WorkspaceState): void {
		reseedIds(state);
		const migrate = (node: LayoutNode): void => {
			if (node.kind === 'split') node.children.forEach(migrate);
			else if (node.panelType === 'errors') node.panelType = 'console';
		};
		for (const ws of state.workspaces) migrate(ws.root);
	}

	/** Apply a layout restored from a `.gfi` patch (or any external source). */
	hydrate(state: unknown): void {
		if (!isValidState(state)) return;
		this._adopt(state);
		this.state = state;
		// The manager's own arrangement coming back at us (hello / a loaded patch). Pushing the
		// re-seeded ids back is an echo, not an edit — classifying it as authoring is what
		// re-dirtied a patch moments after it was saved.
		this._replaced();
		this._focusFirst(this.active.root);
	}

	/**
	 * Apply a PEER's authored arrangement — the manager's `layout` event, which fires when another
	 * client splits a panel, adds a tab, picks a viewer kind. The layout is not a CRDT doc root, so
	 * this event is the only way it travels between live clients.
	 *
	 * A merge, not a `hydrate`. The blob is opaque and carries two different things: the peer's
	 * panel TREE, which is shared, and wherever that peer happened to be LOOKING when it authored,
	 * which is not. Replacing wholesale would climb a phone out of the sub-patch it is three levels
	 * into and drag it onto whichever layout tab the desktop had in front — the same "navigation is
	 * not authoring" line the dirty taxonomy draws, on the other side of the wire. So this takes the
	 * structure and keeps this client's viewpoint: the front tab, each surviving panel's sub-patch
	 * depth, and the focused/maximized panel while it still exists.
	 *
	 * A panel we have never seen has no viewpoint of ours to keep, so it arrives as the peer left
	 * it — a split made from inside a sub-patch opens there on both screens rather than diverging.
	 */
	applyRemoteLayout(remote: unknown): void {
		if (!isValidState(remote)) return;
		this._adopt(remote);

		const myPath = new Map<string, unknown>();
		for (const w of this.state.workspaces) {
			for (const p of collectPanels(w.root)) myPath.set(p.id, asStateObject(p.state).subpatchPath);
		}
		const keepMyPath = (node: LayoutNode): void => {
			if (node.kind === 'split') {
				node.children.forEach(keepMyPath);
				return;
			}
			if (!myPath.has(node.id)) return;
			const path = myPath.get(node.id);
			const state = asStateObject(node.state);
			node.state =
				path === undefined
					? Object.fromEntries(Object.entries(state).filter(([k]) => k !== 'subpatchPath'))
					: { ...state, subpatchPath: path };
		};
		for (const w of remote.workspaces) keepMyPath(w.root);

		const mine = this.state.activeWorkspaceId;
		this.state = {
			...remote,
			activeWorkspaceId: remote.workspaces.some((w) => w.id === mine) ? mine : remote.activeWorkspaceId
		};
		this._replaced(); // a peer's arrangement is nobody's local edit
		this._remoteApplied = true; // …and must not be pushed back (AFTER `_replaced`, which clears it)

		// Focus and maximize are viewpoint too — leave them where they are unless the peer closed
		// the panel they name. (Ids are never reused, so a stale one can only ever miss.)
		const root = this.active.root;
		if (!this.activePanelId || !findPanel(root, this.activePanelId)) {
			this.activePanelId = firstPanelId(root);
		}
		if (this.maximizedPanelId && !findPanel(root, this.maximizedPanelId)) this.maximizedPanelId = null;
	}

	/** Run a tracked layout mutation `fn`, recording one history entry iff it
	 * actually changed the state tree (a no-op split / blocked close records
	 * nothing). `coalesceKey` merges a continuous gesture into one entry.
	 * Undoable ⇒ authored: every mutation that earns a history entry is, by
	 * definition, the user editing the arrangement. (The intent is marked even
	 * while history is suspended — a redo still changes the saved layout.) */
	private _tracked(kind: LayoutActionKind, label: string, fn: () => void, coalesceKey?: string): void {
		const before = this.serialize();
		const prev = this.state;
		fn();
		if (this.state === prev) return;
		this._mark('authored');
		if (!history().isSuspended) {
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

	/** Returns whether the tree actually changed, so a caller can classify only real writes. */
	private _updateActiveRoot(fn: (root: LayoutNode) => LayoutNode | null): boolean {
		const ws = this.active;
		const root = fn(ws.root);
		if (!root || root === ws.root) return false;
		this._setRoot(ws.id, root);
		return true;
	}

	/** Like `_updateActiveRoot`, for a write addressed by PANEL id rather than by "what is in
	 * front": panel ids are unique across tabs and every tree op is a no-op on a tree lacking the
	 * id, so mapping `fn` over all of them lands it wherever the panel lives. The capture and the
	 * clearing of node bindings already walk every tab (`panelsBoundTo` / `clearNodeRefs`), so the
	 * undo restore has to as well — through the active root alone it silently dropped the re-bind
	 * of any panel sitting in a background tab. */
	private _updateAnyRoot(fn: (root: LayoutNode) => LayoutNode | null): boolean {
		let changed = false;
		const workspaces = this.state.workspaces.map((w) => {
			const root = fn(w.root);
			if (!root || root === w.root) return w;
			changed = true;
			return { ...w, root };
		});
		if (!changed) return false;
		this.state = { ...this.state, workspaces };
		return true;
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

	/** `containerPx` is the split's measured size along its axis — the denominator of the pixel
	 * floor (D-R10). The splitter already measures it to convert px into a fraction; passing it on
	 * is what lets the model floor a panel at a size rather than at a percentage. */
	resize(splitId: string, dividerIndex: number, delta: number, containerPx = 0): void {
		// A splitter drag fires this per mousemove — coalesce the whole drag into
		// one undo step via a stable per-divider key.
		this._tracked(
			'resize_split',
			'Resize',
			() =>
				this._updateActiveRoot((root) =>
					resizeSplit(root, splitId, dividerIndex, delta, containerPx)
				),
			`resize:${splitId}:${dividerIndex}`
		);
	}

	setType(panelId: string, panelType: string): void {
		this._tracked('set_panel_type', 'Change panel', () => {
			this._updateActiveRoot((root) => setPanelType(root, panelId, panelType));
		});
	}

	/** Write a panel's opaque state WITHOUT a history entry of its own — for a write that is either
	 * not an edit (`'navigation'`: the sub-patch path) or whose undo step another domain already
	 * owns (the view domain's `set_view`, which carries a finer kind/settings payload than a layout
	 * snapshot, and its own replay). Every OTHER authored panel write must be `_tracked`: layout
	 * undo restores a whole `WorkspaceState`, so an unrecorded write landing after a tracked action
	 * is in neither of its snapshots and the undo destroys it. That is what the two ops below are. */
	setPanelState(panelId: string, state: unknown, intent: LayoutIntent = 'authored'): void {
		if (this._updateAnyRoot((root) => setPanelState(root, panelId, state))) this._mark(intent);
	}

	/** Merge `patch` into a panel's state bag as one tracked, undoable edit. The shared body of the
	 * authored panel-state ops — each names itself so the undo button reads like the click. */
	private _patchPanelState(
		kind: LayoutActionKind,
		label: string,
		panelId: string,
		patch: (state: unknown) => Record<string, unknown>
	): void {
		this._tracked(kind, label, () => {
			this._updateAnyRoot((root) => {
				const p = findPanel(root, panelId);
				if (!p) return root;
				return setPanelState(root, panelId, patch(p.state));
			});
		});
	}

	setActive(panelId: string): void {
		if (this.activePanelId !== panelId) this.activePanelId = panelId;
	}

	/** Bind a node to a linkable panel (merges `node` into its state, keeping
	 * any slot/kind). Called when a node is dragged onto the panel. */
	linkNodeToPanel(panelId: string, nodeName: string): void {
		this._patchPanelState('link_node_to_panel', 'Bind node to panel', panelId, (s) =>
			withLinkedNode(s, nodeName)
		);
	}

	/** Release a linkable panel's bound node — the ✕ in NodeLinkedPanel's bar and ConsolePanel's
	 * filter chip. The exact inverse of `linkNodeToPanel`, and tracked for the same reason. */
	unlinkNodeFromPanel(panelId: string): void {
		this._patchPanelState('unlink_node_from_panel', 'Unbind node from panel', panelId, (s) =>
			withLinkedNode(s, null)
		);
	}

	/** Pick the output slot a Viewer / Metadata panel reads from its bound node. */
	setPanelSlot(panelId: string, slot: string): void {
		this._patchPanelState('set_panel_slot', 'Select slot', panelId, (s) => ({
			...asStateObject(s),
			slot
		}));
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
		if (!changed) return;
		this.state = { ...this.state, workspaces };
		this._mark('authored'); // the node delete that caused it already changed the patch
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

	/** D-R11: switching layout tabs is NAVIGATION. It changes which arrangement is in front, not
	 * what any panel holds — the same "looking elsewhere" as entering a sub-patch, and the move
	 * `navContext` makes to re-orient an undo. Creating, renaming, reordering or closing a tab is
	 * still authoring; only the selection is a look. */
	selectTab(workspaceId: string): void {
		if (this.state.activeWorkspaceId === workspaceId) return;
		const ws = this.state.workspaces.find((w) => w.id === workspaceId);
		if (!ws) return;
		this.state = { ...this.state, activeWorkspaceId: workspaceId };
		this._mark('navigation');
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
				id: uid('ws'),
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
