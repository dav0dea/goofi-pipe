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
	findPanel,
	firstPanelId,
	resizeFractions,
	type Direction,
	type LayoutNode,
	type Workspace,
	type WorkspaceState
} from './model';
import {
	buildWorkspaces,
	childIds,
	firstPanelIn,
	pageOf,
	splitFractions,
	type Arrangement
} from './arrangement';
import { asStateObject } from './panelState';
import { history } from '$lib/stores/history.svelte';
import { captureNavContext } from './navContext';
import { getControl, type Control } from '$lib/api/control';
import type { OpName } from '$lib/api/ops';

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
 *   `page_set_panel` command: undoable, converged to every peer, and it dirties.
 * - `'navigation'` — the user only changed what they are LOOKING at (entering a sub-patch). It
 *   becomes viewpoint: stored for this client, never converged, never dirtying.
 *
 * The routing follows the WRITE, never the device — phone and desktop share this one rule.
 */
export type LayoutIntent = 'navigation' | 'authored';

/** What `session_add_page` answers: the ids it minted, which no client can otherwise know. */
interface Page {
	page: string;
	panel: string;
}

/** What `set_viewpoint` stores for this client, and what a reload gets back. */
export interface Viewpoint {
	page?: string;
	panel?: string;
	paths?: Record<string, string>;
}

class WorkspaceStore {
	/** The manager's arrangement, mirrored from the doc root. Read-only: a write is a command. */
	private _arr = $state<Arrangement>({});
	/** The shares a splitter drag is currently drawing, before it commits. A resize is one
	 * continuous gesture, so the override lives here for its duration and lands as ONE
	 * `page_resize_split` on pointer-up — never a command per pointermove. */
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
	/** The page and root panel a just-accepted `session_add_page` minted, brought forward once the
	 * doc catches up. The manager answers with the ids before the CRDT delta carrying them arrives,
	 * so they are known first but cannot be drawn yet. */
	private _wantPage: { page: string; panel: string } | null = null;
	/** Page names claimed but not yet seen in the replica — see `_claimName`. */
	private _claimed = new Set<string>();
	/** Last-focused panel id — keyboard shortcuts scope to this. */
	activePanelId = $state<string | null>(null);
	/** When set, only this panel renders, filling the workspace. */
	maximizedPanelId = $state<string | null>(null);
	/** The panel or tab currently being dragged. While set, panels show edge drop zones and the tab
	 * bar accepts the drop. */
	dragging = $state<DragRef | null>(null);
	/** Bumped whenever the viewpoint changes, so the shell can persist it debounced. */
	viewpointEpoch = $state(0);
	/** How a layout command reaches the manager. Defaults to the live socket; a test injects a
	 * double, the way `history.configureDeps` already does. */
	private _control: () => Control = getControl;

	/** Test seam: send layout commands through an injected control. */
	configureControl(provider: () => Control): void {
		this._control = provider;
	}

	/** The arrangement as drawn: the manager's, with the in-flight resize applied. */
	private _effective = $derived.by(() => {
		const d = this._drag;
		if (!d) return this._arr;
		const kids = childIds(this._arr, d.split);
		if (kids.length !== d.sizes.length) return this._arr;
		const next: Arrangement = { ...this._arr };
		kids.forEach((id, i) => (next[id] = { ...next[id], size: d.sizes[i] }));
		return next;
	});

	private _workspaces = $derived.by(() => {
		const paths = this._paths;
		const overlay = (n: LayoutNode): LayoutNode => {
			if (n.kind === 'split') return { ...n, children: n.children.map(overlay) };
			const path = paths[n.id];
			return path === undefined ? n : { ...n, state: { ...asStateObject(n.state), subpatchPath: path } };
		};
		return buildWorkspaces(this._effective).map((w) => ({ ...w, root: overlay(w.root) }));
	});

	/** The tree the panel system renders. Derived, not held — the manager's copy is the state. */
	state = $derived<WorkspaceState>({
		workspaces: this._workspaces,
		activeWorkspaceId: this.active.id
	});

	get active(): Workspace {
		const all = this._workspaces;
		return all.find((w) => w.id === this._page) ?? all[0];
	}

	// --- the replica ---------------------------------------------------------

	/** Adopt the arrangement the manager mirrored, and prune any viewpoint it invalidated — a panel
	 * WE focused that a peer just closed, a page that went with it. This is also where an in-flight
	 * resize override retires: the split's own shares moved, so the drawn tree is the manager's
	 * again. */
	syncFromDoc(arr: Arrangement): void {
		const prev = this._arr;
		this._arr = arr;

		const s = this._sent;
		if (s) {
			const before = splitFractions(prev, s.split);
			const after = splitFractions(arr, s.split);
			if (before.length !== after.length || before.some((v, i) => v !== after[i])) {
				this._sent = null;
				if (!this._dragLive) this._drag = null;
			}
		}
		const want = this._wantPage;
		if (want && arr[want.page]) {
			this._wantPage = null;
			this._page = want.page;
			this._focus(want.panel);
		}
		// A claim's job is over the moment the name it reserved shows up in the replica.
		if (this._claimed.size > 0) for (const w of this._workspaces) this._claimed.delete(w.name);
		if (this._page !== null && !arr[this._page]) this._page = null;
		const root = this.active?.root;
		if (!root) return;
		if (!this.activePanelId || !findPanel(root, this.activePanelId)) {
			this.activePanelId = firstPanelId(root);
		}
		if (this.maximizedPanelId && !arr[this.maximizedPanelId]) this.maximizedPanelId = null;
		for (const id of Object.keys(this._paths)) {
			if (!arr[id]) delete this._paths[id];
		}
	}

	/** Restore the viewpoint this client last stored (it rides the `.gfi` and the snapshot, but is
	 * never converged). Ids that no longer exist are simply not adopted. */
	restoreViewpoint(vp: unknown): void {
		const v = vp as Viewpoint | null;
		if (!v || typeof v !== 'object') return;
		if (typeof v.page === 'string') this._page = v.page;
		if (typeof v.panel === 'string') this.activePanelId = v.panel;
		if (v.paths && typeof v.paths === 'object') this._paths = { ...v.paths };
	}

	/** What `set_viewpoint` stores. Plain JSON: the shell pushes it debounced. */
	viewpoint(): Viewpoint {
		return {
			page: this.active?.id,
			panel: this.activePanelId ?? undefined,
			paths: $state.snapshot(this._paths)
		};
	}

	private _viewpointChanged(): void {
		this.viewpointEpoch += 1;
	}

	// --- commands ------------------------------------------------------------

	/** The page NAME an entry lives on — how every page op addresses it. */
	private _pageName(id: string): string | null {
		const page = pageOf(this._arr, id);
		return page === null ? null : (this._arr[page].name ?? null);
	}

	/** Send one layout command and record ONE undo step for it. The manager captured the exact
	 * inverse, so the client entry only marks the step and delegates — the same contract every
	 * graph mutation has used since the unified command API. A refusal (closing a page's last
	 * panel) records nothing, so the two stacks stay 1:1. */
	private async _cmd<T>(
		label: string,
		op: OpName,
		payload: Record<string, unknown>
	): Promise<T | null> {
		try {
			const res = await this._control().call<T>(op, payload);
			if (!history().isSuspended) {
				history().record({ kind: 'graph_cmd', domain: 'graph', label, context: captureNavContext() });
			}
			return res;
		} catch (e) {
			console.warn(`${op} refused`, e);
			return null;
		}
	}

	/** After a structural layout change, drop the maximized view and focus `panelId`. */
	private _focus(panelId: string): void {
		this.maximizedPanelId = null;
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
		const page = this._pageName(panelId);
		if (!page) return;
		void this._cmd<string>('Split panel', 'page_split_panel', {
			page,
			panel: panelId,
			direction,
			place_before: placeBefore,
			ratio: fraction
		}).then((fresh) => {
			if (typeof fresh === 'string') {
				this.activePanelId = fresh;
				this._viewpointChanged();
			}
		});
	}

	close(panelId: string): void {
		const page = this._pageName(panelId);
		if (page) void this._cmd('Close panel', 'page_remove_panel', { page, panel: panelId });
	}

	/** A splitter drag fires this per pointermove. It draws locally — `containerPx` is the split's
	 * measured size along its axis, the denominator of the pixel floor (D-R10) — and nothing leaves
	 * the client until `commitResize`. */
	resize(splitId: string, dividerIndex: number, delta: number, containerPx = 0): void {
		const base =
			this._drag?.split === splitId ? this._drag.sizes : splitFractions(this._arr, splitId);
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
		const page = this._pageName(splitId);
		// What was last SENT for this split, falling back to the replica. Comparing against the
		// replica alone would drop a second drag that returns the split to its pre-commit shares,
		// because the replica is still showing exactly those.
		const before =
			this._sent?.split === splitId ? this._sent.sizes : splitFractions(this._arr, splitId);
		const same = before.length === d.sizes.length && before.every((s, i) => s === d.sizes[i]);
		if (!page || same) {
			drop();
			return;
		}
		this._sent = { split: splitId, sizes: d.sizes };
		void this._cmd('Resize', 'page_resize_split', { page, split: splitId, fractions: d.sizes }).then(
			(ok) => {
				// A refusal never landed, so it is not a baseline either.
				if (ok === null) {
					this._sent = null;
					drop();
				}
			}
		);
	}

	setType(panelId: string, panelType: string): void {
		const page = this._pageName(panelId);
		if (page) void this._cmd('Change panel', 'page_set_panel', { page, panel: panelId, type: panelType });
	}

	/**
	 * Write a panel's opaque state. `intent` routes it, and that routing IS the dirty taxonomy:
	 * `'navigation'` (the sub-patch path) stays viewpoint and never leaves as a layout op, while an
	 * authored write becomes `page_set_panel` — one command, one undo step, converged to peers.
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
		const page = this._pageName(panelId);
		if (!page) return;
		// The sub-patch path is viewpoint and must not ride a shared write.
		const { subpatchPath: _drop, ...shared } = bag;
		void this._cmd(label, 'page_set_panel', { page, panel: panelId, state: shared });
	}

	setActive(panelId: string): void {
		if (this.activePanelId === panelId) return;
		this.activePanelId = panelId;
		this._viewpointChanged();
	}

	/* The three panel-state edits below each name ONLY the key they change: `page_set_panel` merges,
	 * so reading the bag back first would buy nothing and cost the class the merge exists to kill —
	 * a second write inside the first's round trip replacing a bag it never saw the first land in.
	 * Addressing is `setPanelState`'s, which resolves the page from the whole arrangement, so a panel
	 * on a page in the background is written just like one in front. */

	/** Bind a node to a linkable panel. Called when a node is dragged onto the panel. */
	linkNodeToPanel(panelId: string, nodeUid: string): void {
		this.setPanelState(panelId, { node: nodeUid }, 'authored', 'Bind node to panel');
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
		this.maximizedPanelId = this.maximizedPanelId === panelId ? null : panelId;
		this._viewpointChanged();
	}

	// --- tabs --------------------------------------------------------------

	/** Claim a free page name. A page is addressed BY NAME, so the manager refuses a duplicate — and
	 * the replica does not update until the round trip lands, so a gesture repeated faster than that
	 * (six taps on ＋) would ask for the same free name six times and have five refused. The claim is
	 * therefore remembered until the name is seen, or until the op it was claimed for is refused. */
	private _claimName(base: string): string {
		const names = new Set([...this._workspaces.map((w) => w.name), ...this._claimed]);
		let name = base;
		for (let i = 2; names.has(name); i += 1) name = `${base} ${i}`;
		this._claimed.add(name);
		return name;
	}

	/** Add a layout page. `panelType` is the agent façade's door onto "a tab showing X" — the tab
	 * bar's own + button births the default editor. Two ops when it is given, grouped as one client
	 * entry whose two children pop the two manager commands in order. */
	addTab(panelType?: string): void {
		const name = this._claimName('Layout');
		void history().transaction('Add tab', async () => {
			const born = await this._cmd<Page>('Add tab', 'session_add_page', { name });
			if (born === null) this._claimed.delete(name);
			if (born) this._wantPage = born;
			if (born && panelType && panelType !== DEFAULT_PANEL_TYPE) {
				await this._cmd('Change panel', 'page_set_panel', {
					page: name,
					panel: born.panel,
					type: panelType
				});
			}
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
		const from = this._arr[workspaceId]?.name;
		if (!trimmed || !from || trimmed === from) return;
		void this._cmd('Rename tab', 'session_rename_page', { from, to: trimmed });
	}

	closeTab(workspaceId: string): void {
		const name = this._arr[workspaceId]?.name;
		if (!name) return;
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
		void this._cmd('Close tab', 'session_remove_page', { name });
	}

	reorderTab(fromIndex: number, toIndex: number): void {
		const name = this._workspaces[fromIndex]?.name;
		if (name === undefined || toIndex < 0 || toIndex >= this._workspaces.length) return;
		void this._cmd('Reorder tabs', 'session_reorder_page', { name, to_index: toIndex });
	}

	/** The id of the subtree a drag names: a panel is a subtree of one, a tab drag carries the
	 * page's whole root. Null when the drag's source has gone. */
	private _subtreeOf(d: DragRef): string | null {
		if (d.kind === 'panel') return this._arr[d.panelId] ? d.panelId : null;
		return childIds(this._arr, d.workspaceId)[0] ?? null;
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
		const page = this._pageName(targetPanelId);
		if (!subtree || !page || subtree === targetPanelId) return;
		// A drop lands on a page already in front, so only the FOCUS moves — onto the panel the user
		// just carried there, which is the one they are now working in.
		const focus = firstPanelIn(this._arr, subtree);
		void this._cmd('Move panel', 'page_insert_at_panel', {
			page,
			subtree,
			target: targetPanelId,
			direction,
			place_before: placeBefore,
			ratio: 0.5
		}).then((ok) => {
			if (ok !== null && focus !== null) this._focus(focus);
		});
	}

	/** Drop the dragged panel onto the tab bar at `index` — it becomes a new tab, built AROUND the
	 * panel it carries. (Tabs dropped on the bar reorder instead — see reorderTab.) */
	dropPanelOnTabBar(index: number): void {
		const d = this.dragging;
		this.dragging = null;
		if (!d || d.kind !== 'panel') return;
		const subtree = this._subtreeOf(d);
		if (!subtree) return;
		const name = this._claimName('Layout');
		void this._cmd<Page>('Move panel to new tab', 'session_add_page', {
			name,
			index,
			subtree
		}).then((born) => {
			if (born === null) this._claimed.delete(name);
			else if (born) this._wantPage = born;
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
