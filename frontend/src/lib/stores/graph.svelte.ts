/**
 * Central reactive graph state, backed by the control WS.
 *
 * Uses Svelte 5 runes — components subscribe by importing this store and
 * reading its `$state` fields directly. The store owns the only writes
 * (driven by control events) so consumers never have to merge.
 */
import {
	getControl,
	paramValues,
	type Control,
	type ControlEvent,
	type DirListing,
	type GraphSnapshot,
	type InstanceInfo,
	type LinkInfo,
	type NodeInstanceInfo,
	type NodeTypeInfo
} from '$lib/api/control';
import { ui } from './ui.svelte';
import { consoleStore } from './console.svelte';
import { selection } from './selection.svelte';
import { workspace } from '$lib/workspace/workspace.svelte';
import type { WorkspaceState } from '$lib/workspace/model';
import { seedInlineView, forgetInlineView, rawInlineView } from '$lib/viewers/inlineView.svelte';
import { resolveKind } from '$lib/viewers/kind';
import type { SettingsMap } from '$lib/viewers/settingsSchema';
import type { ViewerKind } from '$lib/viewers/kind';
import { history, type Action } from './history.svelte';
import { captureNavContext } from '$lib/workspace/navContext';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { SyncClient } from '$lib/crdt/syncClient';
import {
	linkViews,
	nodeViews,
	instanceViews,
	docParams,
	viewersJson,
	globalViews,
	type GlobalView,
	type GlobalType
} from '$lib/crdt/graphDoc';
import { assembleNode, type RuntimeOverlay } from '$lib/crdt/nodeAssembly';
import { assembleInstances, instanceError } from '$lib/crdt/instanceAssembly';
import type { StringParam } from '$lib/api/types';
import type * as Y from 'yjs';

/** Safety net: if a node never reports a ⟳ refresh done (it crashed mid-scan, or
 * the option list was so trivially unchanged the push was coalesced away), lift the
 * spinner after this long so the entry can never stay disabled forever. Generous —
 * an LSL resolve blocks the node's ctrl thread ~4s. */
const REFRESH_SPINNER_TIMEOUT_MS = 15000;

/** Stable key for an in-flight param refresh. U+001F (unit separator) can't occur
 * in a uid/group/name, so it composes them unambiguously. */
function refreshKey(node: string, group: string, name: string): string {
	return `${node}\u001f${group}\u001f${name}`;
}

export class GraphStore {
	nodes = $state<NodeInstanceInfo[]>([]);
	links = $state<LinkInfo[]>([]);
	/** Sub-patch instances keyed by instance id (flatten-at-runtime group nodes). */
	instances = $state<Record<string, InstanceInfo>>({});
	savePath = $state<string | null>(null);
	unsavedChanges = $state(false);
	connected = $state(false);
	hadHello = $state(false);

	/** Patch globals (system + user), doc-authoritative, in system-first/creation order. Derived from
	 * the CRDT `globals` root on every doc transaction; the Globals panel reads + edits this. */
	globals = $state<GlobalView[]>([]);

	/** Bumps on every *wholesale* graph load (hello / graph_replaced), never on
	 * incremental node add/remove. Editor panels watch it to fit the view to a
	 * freshly-loaded patch — without auto-fitting when nodes are placed one by
	 * one (which is what made the first interactively-placed node jump). */
	loadEpoch = $state(0);

	nodeTypes = $state<NodeTypeInfo[] | null>(null);

	/** Params with a ⟳ refresh in flight, keyed by `refreshKey` → its safety-timeout
	 * handle. Reactive so a param widget disables its control + shows a spinner while
	 * present. A refresh completes asynchronously (the node re-scans off its ctrl
	 * thread and pushes fresh options on a later `state_update`), so this is set when
	 * the RPC is dispatched and cleared when the node reports the param done
	 * (`refreshed_params`) — not on the fire-and-forget RPC ack. */
	private _refreshing = $state<Record<string, ReturnType<typeof setTimeout>>>({});

	/** instance_id of the manager process we last hydrated from. A change means
	 * the backend was restarted under our still-open tab — a fresh session,
	 * not a transient reconnect — so a layout-less snapshot must reset the
	 * layout instead of preserving the previous session's arrangement. */
	private _lastInstanceId: string | null = null;

	/** The last snapshot's per-node runtime overlay, consumed by `_seedRuntime` as each node
	 * materializes from the doc (the doc syncs *after* the snapshot event). */
	private _snapshotRuntime: GraphSnapshot['runtime'] = {};

	/** The control client (injectable for tests; defaults to the live WS one). */
	private ctl: Control;

	/** The CRDT sync driver — the browser replica of the manager's control-plane doc. Phase 2:
	 * `links` are READ from the doc (the retired `link_added`/`link_removed` events no longer
	 * drive them); other subtrees migrate onto the doc one at a time. */
	private _sync: SyncClient;

	constructor(ctl: Control = getControl()) {
		this.ctl = ctl;
		ctl.onConnect((c) => (this.connected = c));
		ctl.on((ev) => this._handle(ev));
		// Mount the CRDT replica and source doc-owned subtrees from it. The manager mirrors every
		// control mutation into the doc and syncs the delta, so a doc transaction (local seed or
		// remote delta) re-derives the reactive state for each subtree cut over to the doc.
		this._sync = new SyncClient(ctl);
		// Register via onDocChange (not doc.on directly) so the observer follows the doc across a
		// reset() — a fresh backend session swaps in a new empty doc.
		this._sync.onDocChange((txn: Y.Transaction) => {
			if (txn.changed.size > 0) this._syncFromDoc();
		});
		this._sync.start();
	}

	/** The CRDT control-plane document (the SSOT clients read; exposed for doc-driven reads). */
	get doc(): Y.Doc {
		return this._sync.doc;
	}

	/** Re-derive every doc-owned subtree from the CRDT doc (Phase 2 read-path cutover). Runs on
	 * every doc transaction (remote delta or local seed). Subtrees migrate here one at a time;
	 * runtime state (error/stage/ufreq) and catalog metadata (slots/category) stay event-sourced. */
	private _syncFromDoc(): void {
		const doc = this._sync.doc;
		// links: the whole set is replaced from the doc.
		this.links = linkViews(doc);
		// globals: the whole set is replaced from the doc (system-first, then user).
		this.globals = globalViews(doc);
		// The catalog is always present in production (it rides on `hello`), so the doc is authoritative
		// for node AND sub-patch identity: build `this.nodes` + `this.instances` from the doc (+ catalog
		// + runtime). Existence/type/name/pos/param value+expr and the whole sub-patch forest come from
		// the doc; descriptors from the catalog; runtime (error/stage/…) stays event-sourced. Both
		// reconcilers self-guard on an absent catalog (the pre-`hello` window) → they no-op until it
		// lands, then `_replaceSnapshot`/`_refreshNodeTypes` rebuild from the doc.
		this._reconcileNodesFromDoc();
		this._reconcileInstancesFromDoc();
	}

	/** Apply a wholesale snapshot. Returns whether it came from a *new* backend
	 * session (changed `instance_id`) — the caller uses that to decide whether to
	 * re-fit / clear history (a same-session reconnect must leave both alone). */
	private _replaceSnapshot(snap: GraphSnapshot): boolean {
		// The node palette rides on hello/graph_replaced (Phase-2 read cutover) so the doc is
		// authoritative for node identity from the first render — no async `list_nodes` window.
		// Absent → an older backend; keep whatever the async fetch set.
		if (snap.node_types?.length) this.nodeTypes = snap.node_types;
		// The snapshot carries NO graph structure: nodes, links and the sub-patch forest all reach
		// us through the CRDT doc, whose binary sync follows this event and drives `_syncFromDoc`.
		// What it does carry is the runtime overlay — stash it so the reconcile can seed each node
		// as it materializes from the doc (see `_seedRuntime`).
		this._snapshotRuntime = snap.runtime ?? {};
		this.savePath = snap.save_path;
		this.unsavedChanges = snap.unsaved_changes;

		// Layout resolution. A snapshot carrying a layout always drives the panel
		// arrangement (a patch loaded from disk, or the manager echoing what we
		// pushed). A layout-less snapshot is ambiguous: from a *new* backend it
		// means "blank session" (reset to default); from the *same* backend it's
		// a transient reconnect whose layout already matches ours (keep it).
		const freshSession = snap.instance_id !== this._lastInstanceId;
		this._lastInstanceId = snap.instance_id;
		if (snap.layout != null) {
			workspace().hydrate(snap.layout);
		} else if (freshSession) {
			workspace().reset();
		}
		// (History reset happens in `_onWholesaleLoad`, which runs on BOTH a fresh session and an
		// in-session load — a same-session reconnect skips it and keeps its history.)
		return freshSession;
	}

	/** React to a wholesale graph load (a new backend session, or a patch loaded
	 * via the Load button). Fit the view (loadEpoch) and drop the error-console
	 * history, which belongs to the graph that just went away — without this it
	 * would show errors for nodes that no longer exist and could even merge a
	 * reused node name's count across sessions. */
	private _onWholesaleLoad(): void {
		this.loadEpoch += 1;
		// The client's undo/redo stacks are meaningless across a wholesale load: a fresh backend
		// session mints new uids, and an in-session load clears the manager's command history
		// (load_text → CommandHistory::clear), so keeping client entries would pop mismatched. Reset
		// here — this runs on a fresh session AND an in-session load, but NOT a same-session reconnect.
		history().reset();
		consoleStore().clear();
		// Drop stale per-panel selection/inspector state: a loaded layout keeps
		// its saved panel ids, which can collide with ids used earlier this
		// session and silently apply old state to the new panels. (A transient
		// same-session reconnect doesn't come here, so its selection survives.)
		selection().forgetAll();
	}

	/** Seed collapse + kind + settings for a node's slots from its (possibly
	 * restored) `viewers` map. */
	private _seedNodeViewerState(node: NodeInstanceInfo): void {
		const slots = Object.keys(node.output_slots);
		ui().seedNodeViewers(node.uid, slots, node.viewers);
		for (const slot of slots) {
			const v = node.viewers?.[slot];
			seedInlineView(node.uid, slot, {
				kind: v?.kind as ViewerKind | undefined,
				settings: v?.settings as SettingsMap | undefined
			});
		}
	}

	private _viewerPushTimers = new Map<string, ReturnType<typeof setTimeout>>();
	/** Debounced push of a node's full viewer state (collapse / kind / settings) via the
	 * `set_node_viewers` op. The manager stores it (persisted into the .gfi on save) and re-mirrors.
	 * Soft, human-rate view state — NOT a command (not undoable); the debounce keeps it sparse. */
	pushNodeViewers(node: string): void {
		clearTimeout(this._viewerPushTimers.get(node));
		this._viewerPushTimers.set(
			node,
			setTimeout(() => {
				this._viewerPushTimers.delete(node);
				const n = this.nodeById(node);
				if (!n) return;
				const viewers: Record<string, { collapsed: boolean; kind: string; settings: SettingsMap }> = {};
				for (const slot of Object.keys(n.output_slots)) {
					const view = rawInlineView(node, slot);
					viewers[slot] = {
						collapsed: !ui().isSlotExpanded(node, slot),
						kind: resolveKind(n.output_slots[slot], view.kind),
						settings: view.settings
					};
				}
				void this.ctl.call('set_node_viewers', { node, viewers }).catch(() => {
					/* soft view state — a dropped push is harmless (re-pushed on the next edit) */
				});
			}, 250)
		);
	}

	private _handle(ev: ControlEvent): void {
		switch (ev.event) {
			case 'hello': {
				const fresh = this._replaceSnapshot(ev.payload);
				this.hadHello = true;
				// A reconnect to the *same* backend must not re-fit the view or wipe
				// the error history; a new backend session (or first connect) should.
				if (fresh) {
					// A NEW backend session mints uids from 1 again — the stale replica must NOT
					// survive: answering the server's SV would push stale param/pos/expr leaves onto
					// the reused uids (silent corruption), and the reconnect would merge new content
					// into the stale doc (ghost edges). Reset to a fresh empty doc NOW, synchronously,
					// so it happens before this connection answers the server's binary hello SV.
					this._sync.reset();
					this._onWholesaleLoad();
				}
				// The catalog usually rides on the hello snapshot (`_replaceSnapshot`); only fetch it
				// async when an older backend omitted it, so the doc-authoritative path still boots.
				if (!this.nodeTypes?.length) void this._refreshNodeTypes();
				break;
			}
			case 'graph_replaced':
				// A patch was loaded/replaced wholesale — always re-fit + reset history (a load is
				// not undoable across; the manager cleared its history too).
				this._replaceSnapshot(ev.payload);
				this._onWholesaleLoad();
				break;
			// Structure (node existence/name, the sub-patch forest, positions, links) is entirely
			// doc-owned (Phase-2 read cutover): the manager mirrors every group/expand/share/
			// make-unique/add/remove/rename/move into the doc, and `_syncFromDoc`'s doc-reconcile
			// rebuilds `this.nodes` + `this.instances` + `this.links` in place from the delta. The
			// `subpatch_changed` / `node_added` / `node_removed` / `node_renamed` / `boundary_moved` /
			// `node_moved` / `link_added` / `link_removed` events are therefore all retired — the store
			// no longer handles them (the catalog is always present, so there is no fallback window).
			case 'state_update': {
				const t = this.nodeById(ev.payload.node);
				if (t) {
					// Params are doc-owned (the catalog is always present): merge ONLY the runtime bits
					// (expression_error / refreshed options), never wholesale-replace (which would
					// clobber the reconcile's value+descriptor assembly).
					this._mergeParamRuntime(t, ev.payload.params);
					// Lifecycle stage rides every state rebroadcast (authoritative:
					// the manager-side ref derives it from the node's own pushes).
					if (ev.payload.stage) t.stage = ev.payload.stage;
					// The node carries its current error on the state plane (always on,
					// re-pushed): a lost first PROCESSING_ERROR still surfaces here, and a
					// healthy respawn's null clears the stale chip. Applied unconditionally
					// when present so backend truth wins even when no diff-driven `error`
					// event fired.
					if ('error' in ev.payload) t.error = ev.payload.error ?? null;
					// The node advertises its SSE log endpoint here; surfacing it lets
				}
				// A ⟳ refresh finished for these params on this very push (the node
				// re-scanned and now carries fresh options) — lift each spinner exactly
				// when the new options land. Keyed by node id, independent of `t`.
				for (const [group, name] of ev.payload.refreshed_params ?? []) {
					this._endRefresh(refreshKey(ev.payload.node, group, name));
				}
				break;
			}
			case 'node_stage': {
				// Discrete stage transitions the state plane can't carry — today only
				// the terminal bootstrap 'error' (import failure; no auto-restart).
				const t = this._realNode(ev.payload.node);
				if (t) {
					t.stage = ev.payload.stage;
					if (ev.payload.error !== undefined) t.error = ev.payload.error;
				}
				break;
			}
			case 'node_stats': {
				// Low-rate (~1 Hz) self-reported execution telemetry; drives the node's
				// stats overlay + the inspector's stats section. Latest-wins.
				const t = this.nodeById(ev.payload.node);
				if (t) t.stats = ev.payload.stats;
				break;
			}
			case 'param_values': {
				// Live evaluated values of a node's expression-driven params. Applied
				// surgically to the existing descriptors (never a wholesale `params` replace
				// like state_update) so the inspector preview tracks each re-evaluation
				// without clobbering a concurrent edit on a sibling param. These params are
				// in expression mode — never user-editable literals — so there's nothing to
				// race with.
				const t = this.nodeById(ev.payload.node);
				if (t) {
					for (const [group, names] of Object.entries(ev.payload.values)) {
						for (const [name, value] of Object.entries(names)) {
							const p = t.params[group]?.[name];
							// Widen past the discriminated union's narrowed `value` — the
							// backend guarantees the value matches the param's own type.
							if (p) (p as { value: unknown }).value = value;
						}
					}
				}
				break;
			}
			case 'error': {
				// Live snapshot — drives the node's red border, the floating error chip, and the
				// inspector's current-error section. Always on, via the control plane, independent of
				// whether a Console is open. The bridge only ever keys this by a REAL node uid (its
				// error-transition loop iterates node_uids); a collapsed sub-patch's deep error is
				// DERIVED from its members, so after updating the member node we recompute the
				// enclosing instances (this event fires no doc transaction, so the doc-reconcile that
				// normally derives instance error does not run).
				const t = this.nodeById(ev.payload.node);
				if (t) t.error = ev.payload.error;
				if (ev.payload.error)
					consoleStore().ingestError(ev.payload.node, ev.payload.error, Date.now());
				this._recomputeInstanceErrors();
				break;
			}
			case 'unsaved_changes':
				this.unsavedChanges = ev.payload.unsaved_changes;
				break;
			case 'save_path_changed':
				this.savePath = ev.payload.save_path;
				break;
			case 'layout':
				// A patch finished loading after we connected — e.g. CLI startup
				// (`goofi-pipe x.gfi`), whose nodes arrive as node_added events with
				// no graph_replaced snapshot. Hydrate its layout if any (null → keep
				// ours), then fit the view to the freshly-loaded graph. No history
				// reset here: it's the same session, and any load-time errors are
				// the just-loaded patch's own.
				if (ev.payload.layout != null) {
					workspace().hydrate(ev.payload.layout);
					// New layout's panel ids may collide with ones used earlier
					// this session — drop stale per-panel state (see _onWholesaleLoad).
					selection().forgetAll();
				}
				this.loadEpoch += 1;
				break;
		}
	}

	private async _refreshNodeTypes(): Promise<void> {
		try {
			const result = await this.ctl.call<{ types: NodeTypeInfo[] }>('list_nodes');
			this.nodeTypes = result.types;
			// The catalog supplies node descriptors — with it in hand the doc becomes authoritative
			// for node + sub-patch identity, so (re)build both from the doc now (Phase-2 read cutover).
			this._reconcileNodesFromDoc();
			this._reconcileInstancesFromDoc();
		} catch (e) {
			console.warn('list_nodes failed', e);
		}
	}

	// ------------------------------------------------------------------
	// mutations (sent via control RPC; UI updates apply on response)
	// ------------------------------------------------------------------

	/** Push an action onto the history (unless a replay is in progress). */
	private _record(action: Action): void {
		if (!history().isSuspended) history().record(action);
	}

	/** Record ONE graph command on the client history. The manager owns the exact inverse (it
	 * captured the pre-state), so this entry only marks the step — its undo/redo DELEGATE to the
	 * manager's session history (B3) — and carries any client-local layout side-effect (panels
	 * emptied when a node vanished) to re-bind on undo. */
	private _recordGraphCmd(
		label: string,
		boundPanels: Array<{ panelId: string; state: unknown }> = []
	): void {
		this._record({ kind: 'graph_cmd', domain: 'graph', label, context: captureNavContext(), boundPanels });
	}

	async addNode(
		type: string,
		category: string,
		pos: [number, number],
		instId?: string,
		params?: Record<string, Record<string, unknown>>
	): Promise<string> {
		// `instId` lands the node inside that sub-patch (member of the instance);
		// omitted, it goes in the root graph. `params` (paste/duplicate replay) are applied at
		// creation UNDER THE GRAPH LOCK — a post-add leaf-write would no-op until the new node syncs
		// into the replica, silently dropping the values.
		const uid =
			(await this.ctl.call<string>('add_node', { type, category, pos, inst_id: instId, params })) ??
			'';
		// The manager recorded the add (its inverse is a subtree-capturing RemoveNode); mark the step.
		if (uid) this._recordGraphCmd(`Add ${type}`);
		return uid;
	}

	async removeNode(uid: string): Promise<void> {
		// The manager's RemoveNode captures the WHOLE subtree (members, params, links, stubs,
		// membership) for a leaf, a sub-patch member, OR a collapsed instance alike (B3b), so its
		// inverse restores it uid-stably — the client just marks the step and carries the panels the
		// delete will empty (re-bound on undo, since the doc-reconcile won't restore a binding). The
		// label reads the (real or synthetic) node's name before it vanishes.
		const label = `Delete ${this.nodeById(uid)?.name ?? uid}`;
		const boundPanels = history().isSuspended ? [] : workspace().panelsBoundTo(uid);
		await this.ctl.call('remove_node', { node: uid });
		this._recordGraphCmd(label, boundPanels);
	}

	/** Respawn a (typically crashed) node: the backend restarts its process IN PLACE,
	 * preserving the uid (so links/panels reconnect), display name, params, position,
	 * sub-patch membership, and links — and re-wires status forwarding itself. A recovery
	 * action, not a semantic edit, so it records no history. (A remove+add would land a
	 * sub-patch member back at ROOT and, post Bug-C, mirror-remove a SHARED member across
	 * its siblings — restart_node avoids both.) */
	async restartNode(uid: string): Promise<void> {
		await this.ctl.call('restart_node', { node: uid });
	}

	async addLink(link: LinkInfo): Promise<void> {
		// The manager's AddLink captures any wire its single-source rule displaces, so its inverse
		// restores it — the client just marks the step.
		await this.ctl.call('add_link', link as unknown as Record<string, unknown>);
		this._recordGraphCmd('Connect');
	}

	async removeLink(link: LinkInfo): Promise<void> {
		await this.ctl.call('remove_link', link as unknown as Record<string, unknown>);
		this._recordGraphCmd('Disconnect');
	}

	async updateParam(node: string, group: string, name: string, value: unknown): Promise<void> {
		// Guard on the param's EXISTENCE (a real param may hold 0/false/''): a missing param (agent
		// typo, or a call racing node hydration) would otherwise send a bogus edit the manager rejects.
		const param = this.nodeById(node)?.params?.[group]?.[name];
		if (!param) throw new Error(`update_param: no param ${group}.${name} on node ${node}`);
		await this.ctl.call('update_param', { node, group, name, value });
		this._recordGraphCmd(`Set ${name}`);
	}

	// ── Globals mutators ────────────────────────────────────────────────────────────────────────
	// Command ops (EditGlobal / a Compound rename) — undoable, and validated server-side (invalid
	// name / collision / protected-system reject the RPC). Each resolves on success and REJECTS on a
	// server refusal, so callers `await` + catch to surface/undo an invalid edit.

	/** Add a NEW user global. Rejects on an invalid name or a collision (server-validated — a
	 * distinct op from `set_global` so an add can't silently overwrite an existing/system global). */
	async addGlobal(name: string, value: number | string | boolean, type: GlobalType): Promise<void> {
		await this.ctl.call('add_global', { name, value, type });
		this._recordGraphCmd(`Add global ${name}`);
	}

	/** Edit an existing global's value (system or user); the declared type + system flag are kept. */
	async setGlobalValue(name: string, value: number | string | boolean): Promise<void> {
		const type = this.globals.find((g) => g.name === name)?.type;
		if (!type) throw new Error(`set_global: no global ${name}`);
		await this.ctl.call('set_global', { name, value, type });
		this._recordGraphCmd(`Set global ${name}`);
	}

	/** Remove a user global (a system global is refused by the server). */
	async removeGlobal(name: string): Promise<void> {
		await this.ctl.call('remove_global', { name });
		this._recordGraphCmd(`Remove global ${name}`);
	}

	/** Rename a user global (add-new + remove-old as one undo step; refs are not rewritten — a stale
	 * `globals.<old>` throws at eval time, per spec). Rejects on a system global or a collision. */
	async renameGlobal(oldName: string, newName: string): Promise<void> {
		await this.ctl.call('rename_global', { old: oldName, new: newName });
		this._recordGraphCmd(`Rename global ${oldName} → ${newName}`);
	}

	/** Ask a live node to re-evaluate a param's options (device / stream pickers).
	 * Recomputes options only, never the value — so it is NOT an undoable edit; the
	 * fresh list arrives on the node's next state_update. The param is marked
	 * refreshing (disabling its widget with a spinner) until the node reports it done. */
	async refreshParam(node: string, group: string, name: string): Promise<void> {
		const key = refreshKey(node, group, name);
		this._beginRefresh(key);
		try {
			await this.ctl.call('refresh_param', { node, group, name });
		} catch (e) {
			// The RPC only *dispatches* the ctrl message; if even that failed the node
			// will never re-scan or push, so lift the spinner now rather than wait out
			// the safety timeout.
			this._endRefresh(key);
			throw e;
		}
	}

	/** Whether a ⟳ refresh is currently in flight for this param — the widget reads
	 * this to disable its control and show a spinner. */
	isRefreshing(node: string, group: string, name: string): boolean {
		return refreshKey(node, group, name) in this._refreshing;
	}

	private _beginRefresh(key: string): void {
		this._endRefresh(key); // coalesce a rapid re-click onto one in-flight refresh
		const handle = setTimeout(() => this._endRefresh(key), REFRESH_SPINNER_TIMEOUT_MS);
		this._refreshing = { ...this._refreshing, [key]: handle };
	}

	private _endRefresh(key: string): void {
		const handle = this._refreshing[key];
		if (handle === undefined) return;
		clearTimeout(handle);
		const { [key]: _drop, ...rest } = this._refreshing;
		this._refreshing = rest;
	}

	async setExpression(
		node: string,
		group: string,
		name: string,
		expression: string | null,
		opts: { enabled?: boolean; triggers_process?: boolean } = {}
	): Promise<void> {
		const d = this.nodeById(node)?.params?.[group]?.[name];
		// Guard on the param's EXISTENCE (like updateParam): a missing param (agent typo, or a call
		// racing node hydration) would otherwise send a binding for a phantom param the manager rejects.
		if (!d) throw new Error(`set_expression: no param ${group}.${name} on node ${node}`);
		await this.ctl.call('set_expression', {
			node,
			group,
			name,
			expression,
			enabled: opts.enabled ?? false,
			triggers: opts.triggers_process ?? false
		});
		this._recordGraphCmd(`Set ${name} expression`);
	}

	async setNodePos(uid: string, pos: [number, number]): Promise<void> {
		// Committed on drag-stop only; live drag stays local to Svelte Flow. The manager's EditNode
		// captures the prior pos so its inverse restores it. Handles a node OR an instance facade.
		await this.ctl.call('set_node_pos', { node: uid, pos });
		this._recordGraphCmd(`Move ${this.nodeById(uid)?.name ?? uid}`);
	}

	/** Set a node's mutable display name (uid identity is unchanged). */
	async renameNode(uid: string, name: string): Promise<void> {
		const oldName = this.nodeById(uid)?.name ?? '';
		if (oldName === name) return;
		await this.ctl.call('rename_node', { node: uid, name });
		this._recordGraphCmd(`Rename ${oldName} → ${name}`);
	}

	/** Push the current workspace layout into the running patch (manager memory)
	 * so it survives reloads and lands in the .gfi on save. Fire-and-forget:
	 * layout is soft UI state, so a dropped push is harmless. */
	async setLayout(layout: unknown): Promise<void> {
		try {
			await this.ctl.call('set_layout', { layout });
		} catch {
			/* not connected / in flight — ignore */
		}
	}

	async save(
		path?: string,
		overwrite = false,
		layout?: unknown
	): Promise<{ path: string; yaml: string }> {
		// `layout` is the frontend workspace arrangement; the backend writes it
		// into the .gfi. Omitted (undefined) → not sent → backend keeps any
		// existing layout.
		return this.ctl.call('save', { path, overwrite, layout });
	}

	/** Load a patch, replacing the current graph. Loading fully RESETS the session (the manager
	 * clears its command history; the client's stacks reset on the `graph_replaced` wholesale-load),
	 * so there is nothing to undo across it — no history entry (spec §3: no load command). */
	async loadText(content: string): Promise<void> {
		await this.ctl.call('load_text', { content });
	}

	/** Group the named nodes into a sub-patch. Returns its instance id. */
	async groupNodes(members: string[], pos?: [number, number]): Promise<string> {
		const r = await this.ctl.call<{ inst_id: string }>('group_nodes', { members, pos });
		if (r?.inst_id) this._recordGraphCmd('Group nodes');
		return r.inst_id;
	}

	/** Dissolve a sub-patch instance back into its member nodes. */
	async expandInstance(instId: string): Promise<void> {
		await this.ctl.call('expand_instance', { inst_id: instId });
		this._recordGraphCmd('Ungroup');
	}

	/** Add a virtual In/Out boundary node to a sub-patch (unwired). Returns its id. */
	async addBoundary(
		instId: string,
		dir: 'in' | 'out',
		dtype: string,
		pos: [number, number]
	): Promise<string> {
		const r = await this.ctl.call<{ bnd_id: string }>('add_boundary', {
			inst_id: instId,
			dir,
			dtype,
			pos
		});
		if (r?.bnd_id) this._recordGraphCmd('Add boundary');
		return r.bnd_id;
	}

	/** Set (inner_node/slot) or clear (nulls) a boundary's single inner target. */
	async wireBoundary(
		instId: string,
		bndId: string,
		innerNode: string | null,
		innerSlot: string | null
	): Promise<void> {
		await this.ctl.call('wire_boundary', {
			inst_id: instId,
			bnd_id: bndId,
			inner_node: innerNode,
			inner_slot: innerSlot
		});
		this._recordGraphCmd('Wire boundary');
	}

	/** Delete an In/Out boundary node (tears down its external wires). */
	async removeBoundary(instId: string, bndId: string): Promise<void> {
		await this.ctl.call('remove_boundary', { inst_id: instId, bnd_id: bndId });
		this._recordGraphCmd('Remove boundary');
	}

	/** Rename an In/Out portal (its label + the sub-patch's exposed slot name). The routing key
	 * (bndId) is unchanged, so external wires survive. Records AFTER the RPC lands so a rejected
	 * rename (blank/duplicate) doesn't poison history. */
	async renameBoundary(instId: string, bndId: string, name: string): Promise<void> {
		const oldName = this.instances[instId]?.interface?.[bndId]?.name ?? bndId;
		if (name === oldName) return;
		await this.ctl.call('rename_boundary', { inst_id: instId, bnd_id: bndId, name });
		this._recordGraphCmd('Rename boundary');
	}

	/** Move an In/Out pill inside the entered view. */
	async setBoundaryPos(instId: string, bndId: string, pos: [number, number]): Promise<void> {
		await this.ctl.call('set_boundary_pos', { inst_id: instId, bnd_id: bndId, pos });
		this._recordGraphCmd('Move boundary');
	}

	/** List one directory level on the BACKEND filesystem (full FS, no jail). */
	async listDir(path?: string): Promise<DirListing> {
		return this.ctl.call<DirListing>('list_dir', { path });
	}

	/** Load a patch from a BACKEND filesystem path (destructive — replaces the graph). Like
	 * {@link loadText}, a load resets the session, so it is not undoable (no history entry). */
	async load(path: string): Promise<void> {
		await this.ctl.call('load', { path });
	}

	/** Current patch as `.gfi` YAML, without writing to disk (for browser download). */
	async serialize(): Promise<{ yaml: string }> {
		return this.ctl.call<{ yaml: string }>('serialize');
	}
	// ------------------------------------------------------------------
	// reads
	// ------------------------------------------------------------------

	/** A real node by uid (no sub-patch synthesis), or null. */
	private _realNode(uid: string): NodeInstanceInfo | null {
		return this.nodes.find((n) => n.uid === uid) ?? null;
	}

	/** Resolve a node by its UID — the one accessor the rest of the app and the
	 * agent surface use. A sub-patch instance id resolves to a *virtual* node
	 * carrying a `subpatch` marker, so selection / inspector / drag treat a
	 * sub-patch exactly like a node (no node class is instantiated). The synth
	 * node's own `uid` is the instance id, so the identity stays uniform. */
	nodeById(id: string): NodeInstanceInfo | null {
		const real = this._realNode(id);
		if (real) return real;
		// ROOT is a real scope in the mirror (the editor renders the root canvas from its
		// members), but it is the canvas itself — never a selectable/synth group node.
		if (id === ROOT_ID) return null;
		const inst = this.instances[id];
		if (!inst) {
			this._synthCache.delete(id);
			return null;
		}
		return this._synthSubpatchNode(id, inst);
	}

	/** Memoized virtual nodes for sub-patch instances, keyed by instance id. A real
	 * node is one stable `$state` object across `nodeById` calls; without this the
	 * synthesized stand-in was rebuilt every call, so a flowNodes rebuild (which
	 * runs on any selection change) handed each sub-patch a fresh `data.node` and
	 * its inline viewer re-subscribed/flickered. The cache restores that stability. */
	private _synthCache = new Map<string, { sig: string; node: NodeInstanceInfo }>();

	/** Validate that `uid` is a direct member of `instId` and return it (flat model: members are
	 * keyed by uid, so this is an identity-with-membership-check — used to draw a stub's inner edge
	 * only when its `inner_node` is genuinely a member). */
	memberUid(instId: string, uid: string): string | null {
		return this.instances[instId]?.members[uid]?.uid ?? null;
	}

	/** Reconcile the flat node list IN PLACE by uid (mirror of _reconcileInstances):
	 * Object.assign a surviving node's fields so its object reference — and thus its
	 * inline-viewer subscription — stays stable, insert + seed genuinely-new nodes, and
	 * fully forget genuinely-vanished ones. The structural cure for the viewer flicker a
	 * wholesale `this.nodes = snap.nodes` (plus a blanket forgetInlineView over every
	 * node) caused on every group/expand/share/make-unique. */
	private _reconcileNodes(next: NodeInstanceInfo[]): void {
		const byUid = new Map(this.nodes.map((n) => [n.uid, n]));
		const nextUids = new Set(next.map((n) => n.uid));
		for (const old of this.nodes) {
			if (nextUids.has(old.uid)) continue;
			ui().forget(old.uid);
			forgetInlineView(old.uid);
			workspace().clearNodeRefs(old.uid);
		}
		this.nodes = next.map((n) => {
			const cur = byUid.get(n.uid);
			if (cur) {
				// subpatch_changed is a STRUCTURE event (group/expand/share/make-unique):
				// membership/names/instances moved but the live node processes are
				// unchanged. Their runtime lifecycle state (stage/error/stats/restarts/
				// stats) is owned by the state_update / node_* stream, and this
				// snapshot was built on a manager thread a later state_update may have
				// overtaken — so copying its volatile fields would REGRESS them (a ready
				// node flickering back to a boot spinner, a cleared error chip reappearing,
				// live stats blanked). Refresh the structural fields in place; keep the
				// survivor's runtime state, which its authoritative stream owns.
				Object.assign(cur, n, { stage: cur.stage, error: cur.error, stats: cur.stats });
				return cur;
			}
			this._seedNodeViewerState(n); // genuinely new node — seed its inline view state
			return n;
		});
	}

	/** The runtime overlay for a node materializing from the doc for the FIRST time: whatever the
	 * last snapshot reported for that uid. Only the seed — from then on `_extractRuntime` carries
	 * the live, event-sourced state forward. Empty for a node created after the snapshot (its
	 * runtime arrives on the stream). */
	private _seedRuntime(uid: string): RuntimeOverlay {
		const seed = this._snapshotRuntime[uid];
		return seed ? { stage: seed.stage, error: seed.error ?? null } : {};
	}

	/** Pull the RUNTIME (event-sourced, never-in-the-doc) fields off an existing node so a doc
	 * re-assemble preserves them — error/stage/stats/membership at node level, and
	 * per-param expression_error + refreshed StringParam options. */
	private _extractRuntime(node: NodeInstanceInfo): RuntimeOverlay {
		const params: NonNullable<RuntimeOverlay['params']> = {};
		for (const group of Object.keys(node.params)) {
			params[group] = {};
			for (const name of Object.keys(node.params[group])) {
				const p = node.params[group][name];
				const pr: NonNullable<RuntimeOverlay['params']>[string][string] = {
					expression_error: p.expression_error
				};
				if (p.type === 'string') pr.options = (p as StringParam).options;
				// An expression param's DISPLAYED value is the live evaluated one (from param_values),
				// which is never written to the doc. Carry it across the rebuild — else the re-assemble
				// reverts it to the doc's committed literal (the fallback path's `expression_enabled`
				// skip, ported to the doc-authoritative path).
				if (p.expression_enabled) pr.liveValue = p.value;
				params[group][name] = pr;
			}
		}
		// `membership` is intentionally NOT extracted here — the caller (`_reconcileNodesFromDoc`)
		// always re-derives it from the doc's instance forest, so carrying it would be dead.
		return {
			error: node.error,
			stage: node.stage,
			stats: node.stats,
			params
		};
	}

	/** Merge ONLY the runtime param bits (expression_error + refreshed StringParam options) from a
	 * state_update's descriptor map onto the existing node — used when the catalog is authoritative,
	 * so the doc-reconcile's value/descriptor assembly is not clobbered by a wholesale replace. */
	private _mergeParamRuntime(
		t: NodeInstanceInfo,
		params: Record<string, Record<string, unknown>>
	): void {
		for (const [group, names] of Object.entries(params)) {
			for (const [name, desc] of Object.entries(names)) {
				const p = t.params[group]?.[name];
				if (!p) continue;
				const d = desc as { expression_error?: string | null; options?: string[] | null };
				(p as { expression_error: string | null }).expression_error = d.expression_error ?? null;
				if (p.type === 'string') (p as StringParam).options = d.options ?? null;
			}
		}
	}

	/** Derive a node's sub-patch membership from the doc's mirrored scope forest. Flat model:
	 * `members` is keyed by uid, so `local_name` is just the uid (no template locals). ROOT → null. */
	private _membershipFromDoc(uid: string): { instance: string; local_name: string } | null {
		for (const iv of instanceViews(this._sync.doc)) {
			if (uid in iv.members) return { instance: iv.uid, local_name: uid };
		}
		return null;
	}

	/** Build `this.nodes` from the CRDT doc (Phase-2 node-identity read cutover): each real node is
	 * assembled from the doc (existence/type/name/pos/param value+expr) + the catalog descriptor (by
	 * type) + the runtime overlay (kept off the doc). Reuses `_reconcileNodes` so survivors keep
	 * object identity (inline viewers don't re-subscribe) and their runtime, new nodes seed viewer
	 * state, and vanished nodes tear down. Called on every doc transaction and when the catalog
	 * (which supplies descriptors) arrives. */
	private _reconcileNodesFromDoc(): void {
		if (!this.nodeTypes?.length) return; // no catalog yet → keep the current nodes; rebuild when it lands
		const doc = this._sync.doc;
		const next: NodeInstanceInfo[] = nodeViews(doc).map((nv) => {
			const existing = this._realNode(nv.uid);
			const catalog = this.nodeTypes?.find((t) => t.type === nv.type);
			const runtime: RuntimeOverlay = existing ? this._extractRuntime(existing) : this._seedRuntime(nv.uid);
			runtime.membership = this._membershipFromDoc(nv.uid);
			const viewers = (viewersJson(doc, nv.uid) ?? {}) as NodeInstanceInfo['viewers'];
			return assembleNode(nv, docParams(doc, nv.uid), viewers, catalog, runtime);
		});
		this._reconcileNodes(next);
	}

	/** Build `this.instances` (the whole sub-patch forest, INCLUDING the synthetic ROOT scope) from
	 * the CRDT doc (Phase-2 instances read cutover). Every structural field is reconstructed from the
	 * doc to match the backend's `describe_instance`/`root_instance`; `error` is DERIVED from the
	 * members' runtime node errors (the bridge only emits `error` keyed by a real node uid, never an
	 * instance uid — so an instance's deep error must be recomputed, not overlaid). Wraps
	 * `_reconcileInstances` with the same vanished-teardown + seed-only-new lifecycle the
	 * `subpatch_changed` handler applied, so a collapsed sub-patch's live viewer state survives. */
	private _reconcileInstancesFromDoc(): void {
		if (!this.nodeTypes?.length) return; // no catalog yet → keep event-sourced instances
		const doc = this._sync.doc;
		const nodes = nodeViews(doc).map((n) => ({ uid: n.uid, name: n.name }));
		const next = assembleInstances(instanceViews(doc), nodes, (uid) => this._realNode(uid)?.error ?? null);
		// Vanished instances fire no per-node event — clear any panel still bound to one (mirror of the
		// retired `subpatch_changed` wrap; `_reconcileInstances` itself drops the map entry + synth cache).
		const before = new Set(Object.keys(this.instances));
		for (const iid of before) if (!(iid in next)) workspace().clearNodeRefs(iid);
		this._reconcileInstances(next);
		// Seed viewer state for a genuinely-NEW instance's output-boundary slots (its synth node carries
		// the blob) — never a survivor (would clobber its live, un-pushed collapse/kind).
		for (const iid of Object.keys(this.instances)) {
			if (before.has(iid)) continue;
			const sn = this.nodeById(iid);
			if (sn) this._seedNodeViewerState(sn);
		}
	}

	/** Re-derive every instance's deep error from its members' current runtime node errors and apply
	 * it in place. Instance error is DERIVED (never in the doc), and a runtime member `error` event
	 * fires no doc transaction — so the doc-reconcile that normally derives it doesn't run. Called
	 * from the `error` handler so a collapsed sub-patch's badge tracks a member error/recovery live. */
	private _recomputeInstanceErrors(): void {
		if (!this.nodeTypes?.length) return; // instances are event-sourced until the catalog lands
		const views = instanceViews(this._sync.doc);
		const byUid = new Map(views.map((v) => [v.uid, v]));
		const nodeError = (uid: string) => this._realNode(uid)?.error ?? null;
		for (const view of views) {
			const rec = this.instances[view.uid];
			if (!rec) continue;
			const err = instanceError(view, byUid, nodeError);
			if (rec.error !== err) rec.error = err;
		}
	}

	/** Reconcile the instances map IN PLACE by uid: mutate an existing record's fields
	 * (preserving the object reference so the inspector's `$derived inst` and the
	 * synth-node memo stay stable), insert new ones, drop vanished ones. */
	private _reconcileInstances(next: Record<string, InstanceInfo>): void {
		for (const [uid, rec] of Object.entries(next)) {
			const cur = this.instances[uid];
			if (cur) Object.assign(cur, rec);
			else this.instances[uid] = rec;
		}
		for (const uid of Object.keys(this.instances)) {
			if (!(uid in next)) {
				delete this.instances[uid];
				this._synthCache.delete(uid); // cache lifetime tracks the instances map
			}
		}
	}

	/** Build the virtual NodeInstanceInfo that stands in for a sub-patch instance
	 * wherever a node is looked up by name. A sub-patch behaves exactly like a
	 * node: its WIRED boundaries become real input/output slots (so the canvas
	 * renders connectors + output viewers and seeds add-node clicks the same way
	 * as any node — the /data route splices an output boundary to its inner
	 * member's stream). Live sharing state is recomputed by the inspector from
	 * `instances`; the marker just rides the id + glyph/badge bits. */
	private _synthSubpatchNode(instId: string, inst: InstanceInfo): NodeInstanceInfo {
		const error = inst.error ?? null;
		const memberCount = Object.keys(inst.members).length;
		// Signature of everything the synth node RENDERS except position — which is
		// applied to the flow node separately and updated in place below, so a drag
		// (a per-frame pos change) keeps the same identity and never churns the
		// viewer. A selection change touches none of these, so the cache hits.
		// The slots are server-computed, so hashing them covers the wired-port set.
		// Hash the portal names too so a rename re-synthesizes the node's slot labels.
		const labelSig = Object.entries(inst.interface)
			.map(([bid, p]) => `${bid}=${p.name ?? ''}`)
			.join(',');
		const sig = `${inst.name}|${error ?? ''}|${memberCount}|${JSON.stringify(inst.slots)}|${labelSig}`;

		const cached = this._synthCache.get(instId);
		if (cached && cached.sig === sig) {
			cached.node.pos = inst.pos; // keep position fresh without a new identity
			return cached.node;
		}

		// External ports ARE the server-computed slots (a pure passthrough).
		const input_slots: Record<string, string> = { ...inst.slots.input };
		const output_slots: Record<string, string> = { ...inst.slots.output };
		// Exposed ports are keyed by the stable boundary id (the routing handle), but show
		// the portal's renameable NAME — so a rename relabels the collapsed port without
		// re-keying the wire.
		const slot_labels: Record<string, string> = {};
		for (const [bid, port] of Object.entries(inst.interface)) {
			if (port.name) slot_labels[bid] = port.name;
		}
		const node: NodeInstanceInfo = {
			// The synth node's identity IS the instance uid, so `node.uid` is the
			// uniform flow/selection/data key for real and sub-patch nodes alike.
			uid: instId,
			// Display label is the instance's separate name (the uid is opaque).
			name: inst.name ?? instId,
			type: 'Sub-patch',
			category: 'subpatch',
			doc: '',
			input_slots,
			output_slots,
			slot_labels,
			params: {},
			pos: inst.pos,
			viewers: inst.viewers ?? {},
			membership: null,
			error,
			subpatch: { instId, memberCount }
		};
		this._synthCache.set(instId, { sig, node });
		return node;
	}

	// ------------------------------------------------------------------
	// subgraph orchestration (clone / duplicate / paste / batch remove) —
	// shared by the editor's keyboard handlers and the agent command surface
	// ------------------------------------------------------------------

	/** Create a batch of nodes, replay their param values, then wire the given
	 * links with endpoints remapped to the new uids. Returns the original→new
	 * uid map (spec `key` is the original uid, matching the uid link endpoints).
	 * Per-node/-param/-link failures are swallowed so one bad item doesn't abort
	 * the batch. */
	async instantiateNodes(
		specs: {
			key: string;
			type: string;
			category: string;
			pos: [number, number];
			params: Record<string, Record<string, unknown>>;
		}[],
		links: LinkInfo[] = [],
		instId?: string
	): Promise<Record<string, string>> {
		const rename: Record<string, string> = {};
		for (const s of specs) {
			try {
				// `instId` lands the clones inside the sub-patch being edited (members of
				// the instance); omitted, they go in the root graph. Params ride INLINE on add_node
				// (applied under the graph lock) — a post-add updateParam is now a doc leaf-write that
				// no-ops until the new node has synced into the replica, so it would drop the values.
				const newUid = await this.addNode(s.type, s.category, s.pos, instId, s.params);
				rename[s.key] = newUid;
			} catch (e) {
				console.warn('instantiateNodes: add_node failed', e);
			}
		}
		for (const l of links) {
			try {
				await this.addLink({
					node_out: rename[l.node_out] ?? l.node_out,
					node_in: rename[l.node_in] ?? l.node_in,
					slot_out: l.slot_out,
					slot_in: l.slot_in
				});
			} catch {
				/* ignore a link that couldn't be remade */
			}
		}
		return rename;
	}

	/** Duplicate the given nodes by uid (offset from their originals), carrying
	 * their current params and the links among them. Returns the original→new uid
	 * map so the caller can select the clones. */
	async cloneNodes(
		uids: Iterable<string>,
		offset: [number, number] = [40, 40],
		instId?: string
	): Promise<Record<string, string>> {
		const set = new Set(uids);
		// `uids` are node identities (selection sets + flow-node ids are uid-keyed), and link
		// endpoints are uids — so the node filter and the spec key must be uids too,
		// or the rename map in instantiateNodes won't line up with the link remap.
		const nodes = this.nodes.filter((n) => set.has(n.uid));
		if (nodes.length === 0) return {};
		const links = this.links.filter((l) => set.has(l.node_in) && set.has(l.node_out));
		const specs = nodes.map((n) => ({
			key: n.uid,
			type: n.type,
			category: n.category,
			pos: [n.pos[0] + offset[0], n.pos[1] + offset[1]] as [number, number],
			params: paramValues(n)
		}));
		return this.instantiateNodes(specs, links, instId);
	}

	/** Delete several nodes as ONE undoable step.
	 *
	 * Every link touching the batch is removed FIRST as a first-class remove_link
	 * action, then the nodes (which no longer re-capture those links). Because the
	 * transaction replays its children in reverse on undo, this restores every node
	 * BEFORE any link — the only correct order, since a link between two co-deleted
	 * nodes can't be re-added until both endpoints exist again. The old per-node
	 * remove (each re-adding its own links) failed here: the first node's inverse
	 * re-added a link to a sibling that hadn't been re-created yet. */
	async removeNodes(uids: Iterable<string>): Promise<void> {
		const uidList = [...uids];
		if (uidList.length === 0) return;
		const label = `Delete ${uidList.length} node${uidList.length > 1 ? 's' : ''}`;
		// Each removeNode is a manager RemoveNode command that captures its OWN subtree (members +
		// params + links + stubs), so a batch — even one containing collapsed instances — is just a
		// transaction of deletes folded into one undo step. A link into a co-deleted node rides with
		// whichever endpoint owns it, so delete order is immaterial.
		await history().transaction(label, async () => {
			for (const uid of uidList) await this.removeNode(uid);
		});
	}

}

let _store: GraphStore | null = null;
export function graph(): GraphStore {
	if (!_store) _store = new GraphStore();
	return _store;
}
