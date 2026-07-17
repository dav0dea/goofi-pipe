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
	type FsEntry,
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
import { history, type Action, type ExprState } from './history.svelte';
import { captureNavContext } from '$lib/workspace/navContext';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { SyncClient } from '$lib/crdt/syncClient';
import {
	linkViews,
	nodeViews,
	instanceViews,
	paramValue,
	setParamValue,
	setParamExpr as docSetParamExpr,
	setNodePos as docSetNodePos,
	setViewers as docSetViewers
} from '$lib/crdt/graphDoc';
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

	/** A just-recorded `load_patch` action awaiting its post-load layout. The
	 * loaded patch's arrangement settles asynchronously — the `graph_replaced`
	 * echo hydrates it AFTER we record the action — so we stash the payload here
	 * and fill `afterLayout` once that echo lands, letting redo reproduce the
	 * exact post-load layout instead of falling back to the YAML's default. */
	private _pendingLoadAfter: { afterLayout: WorkspaceState | null } | null = null;

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
		this._sync.doc.on('afterTransaction', (txn: Y.Transaction) => {
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
		// positions: node + instance placement is doc-owned (retired `node_moved`). Overlay onto
		// the existing objects so a move re-renders without a wholesale node/instance rebuild.
		for (const nv of nodeViews(doc)) {
			const n = this._realNode(nv.uid);
			if (!n) continue;
			if (n.pos[0] !== nv.pos[0] || n.pos[1] !== nv.pos[1]) n.pos = nv.pos;
			// Committed param values are doc-owned leaves (§4.1). Overlay them so a client's
			// DIRECT leaf-write (apply_client_write) is reflected — the manager does NOT emit a
			// state_update for a doc-write, so this is the only read path for it. Skip
			// expression-enabled params: their displayed value is the LIVE evaluated value
			// (param_values), not the committed literal, so overlaying would flicker it.
			for (const group of Object.keys(n.params)) {
				for (const name of Object.keys(n.params[group])) {
					const p = n.params[group][name];
					if (p.expression_enabled) continue;
					const v = paramValue(doc, nv.uid, group, name);
					if (v !== undefined && p.value !== v) (p as { value: unknown }).value = v;
				}
			}
		}
		for (const iv of instanceViews(doc)) {
			const inst = this.instances[iv.uid];
			if (!inst) continue;
			if (inst.pos[0] !== iv.pos[0] || inst.pos[1] !== iv.pos[1]) inst.pos = iv.pos;
			// Boundary positions are doc-owned too (retired `boundary_moved`).
			for (const b of iv.interface) {
				const port = inst.interface?.[b.bnd_id];
				if (port && (port.pos[0] !== b.pos[0] || port.pos[1] !== b.pos[1])) port.pos = b.pos;
			}
		}
	}

	/** Apply a wholesale snapshot. Returns whether it came from a *new* backend
	 * session (changed `instance_id`) — the caller uses that to decide whether to
	 * re-fit / clear history (a same-session reconnect must leave both alone). */
	private _replaceSnapshot(snap: GraphSnapshot): boolean {
		// Drop ui bookkeeping for any node that's about to disappear, then
		// re-seed viewer state (collapse / kind / settings) for every node.
		for (const old of this.nodes) {
			ui().forget(old.uid);
			forgetInlineView(old.uid);
		}
		for (const n of snap.nodes) this._seedNodeViewerState(n);
		this.nodes = snap.nodes;
		// `links` are sourced from the CRDT doc (Phase 2), not the snapshot — the doc syncs the
		// current link set alongside this hello/graph_replaced echo.
		this._reconcileInstances(snap.instances ?? {});
		// Seed collapse/kind/settings for each instance's output-boundary slots (its
		// synth node carries inst.viewers) so a sub-patch viewer state round-trips.
		for (const iid of Object.keys(this.instances)) {
			const sn = this.nodeById(iid);
			if (sn) this._seedNodeViewerState(sn);
		}
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
		// A new backend session replaced the authoritative world out-of-band —
		// replaying old RPCs is meaningless, so drop the history. The one
		// exception to "never auto-clear" (an in-session load keeps its history
		// as a `load_patch` checkpoint; same instance_id → not fresh).
		if (freshSession) {
			history().reset();
			// History (and any load_patch action awaiting its after-layout) is gone.
			this._pendingLoadAfter = null;
		}
		return freshSession;
	}

	/** React to a wholesale graph load (a new backend session, or a patch loaded
	 * via the Load button). Fit the view (loadEpoch) and drop the error-console
	 * history, which belongs to the graph that just went away — without this it
	 * would show errors for nodes that no longer exist and could even merge a
	 * reused node name's count across sessions. */
	private _onWholesaleLoad(): void {
		this.loadEpoch += 1;
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
	/** Debounced client leaf-write of a node's full viewer state (collapse / kind / settings) into
	 * the CRDT doc (Phase 3 — replaces the `set_node_viewers` RPC). The manager applies it via
	 * `set_node_viewers` (which persists it into the .gfi on save) and re-mirrors. Soft, client-
	 * originated, human-rate UI state — the debounce keeps it well under the CRDT's human-rate tier. */
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
				this._sync.commit((doc) => docSetViewers(doc, node, viewers));
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
				if (fresh) this._onWholesaleLoad();
				void this._refreshNodeTypes();
				break;
			}
			case 'graph_replaced':
				// A patch was loaded/replaced wholesale — always re-fit + reset history.
				this._replaceSnapshot(ev.payload);
				this._onWholesaleLoad();
				// The loaded patch's layout has now settled — stamp it onto the
				// load_patch action we recorded just before this echo, so redo restores
				// the precise post-load arrangement (backlog #20).
				this._captureLoadAfterLayout();
				break;
			case 'subpatch_changed': {
				// Group/expand/share/make-unique rewrote the STRUCTURE, but the live node
				// processes are unchanged — only membership/names/instances moved. Reconcile
				// nodes, links and instances by STABLE uid IN PLACE (never a wholesale array
				// swap) so a surviving node/instance keeps its object identity and its inline
				// viewer never re-subscribes. NOT layout / re-fit — an in-place edit, not a load.
				const snap = ev.payload;
				// Vanished instances fire only this event (no per-node node_removed): clear any
				// panel still bound to one. Vanished member NODES are cleared inside
				// _reconcileNodes; their stable minted uids are never reused, so gone means gone.
				const afterInst = new Set(Object.keys(snap.instances ?? {}));
				for (const iid of Object.keys(this.instances)) {
					if (!afterInst.has(iid)) workspace().clearNodeRefs(iid);
				}
				// Instances present BEFORE this resync — seed viewer state only for the
				// genuinely-new ones (mirror of _reconcileNodes' seed-only-new rule). Re-seeding
				// a SURVIVING instance every subpatch_changed would overwrite its live, un-pushed
				// collapse/kind (seedNodeViewers/seedInlineView clobber unconditionally), re-popping
				// a collapsed viewer open → remounting its ViewerFeed → churning the data sub.
				const knownInst = new Set(Object.keys(this.instances));
				this._reconcileNodes(snap.nodes);
				this.links = snap.links;
				this._reconcileInstances(snap.instances ?? {});
				// Seed collapse/kind/settings for a NEW instance's output-boundary slots (its synth
				// node carries inst.viewers) so a freshly-created sub-patch's viewer state applies.
				for (const iid of Object.keys(this.instances)) {
					if (knownInst.has(iid)) continue;
					const sn = this.nodeById(iid);
					if (sn) this._seedNodeViewerState(sn);
				}
				break;
			}
			// boundary_moved retired: boundary positions are read from the CRDT doc forest
			// (Phase 2, see _syncFromDoc). The manager mirrors every boundary move into the doc.
			case 'node_renamed': {
				// Only the display name changed — the uid (the key everything uses) is
				// stable, so nothing else moves. Just update the label in place.
				const t = this.nodeById(ev.payload.node);
				if (t) t.name = ev.payload.name;
				break;
			}
			case 'node_added':
				// Seed view state for this node's output slots — from the saved
				// patch (`viewers`) if present, else the defaults in the stores.
				this._seedNodeViewerState(ev.payload);
				this.nodes = [...this.nodes.filter((n) => n.uid !== ev.payload.uid), ev.payload];
				// Every node is a member of SOME scope (ROOT for a top-level one). Fold it
				// into the owning instance's members map — the index the canvas renders that
				// scope's children from — so an incremental add keeps the scope's child set
				// in sync without a wholesale subpatch_changed snapshot.
				this._addScopeMember(ev.payload.membership ?? null, ev.payload.uid);
				break;
			case 'node_removed':
				this.nodes = this.nodes.filter((n) => n.uid !== ev.payload.node);
				this.links = this.links.filter(
					(l) => l.node_in !== ev.payload.node && l.node_out !== ev.payload.node
				);
				// Drop it from the owning scope's members map (mirror of node_added) so the
				// scope stays in sync without a snapshot.
				this._removeScopeMember(ev.payload.membership ?? null);
				ui().forget(ev.payload.node);
				forgetInlineView(ev.payload.node);
				consoleStore().forgetNodeDedup(ev.payload.node);
				// Empty any Parameters/Viewer/Metadata panel linked to this node.
				workspace().clearNodeRefs(ev.payload.node);
				break;
			// node_moved retired: node + instance positions are read from the CRDT doc (Phase 2,
			// see _syncFromDoc). The manager mirrors every move into the doc; the sync delta
			// updates the placement without a wholesale rebuild.
			// link_added / link_removed retired: links are read from the CRDT doc (Phase 2). The
			// manager mirrors every link mutation into the doc; the sync delta updates `links`.
			case 'state_update': {
				const t = this.nodeById(ev.payload.node);
				if (t) {
					t.params = ev.payload.params;
					// Lifecycle stage rides every state rebroadcast (authoritative:
					// the manager-side ref derives it from the node's own pushes).
					if (ev.payload.stage) t.stage = ev.payload.stage;
					// A healthy state push means the (possibly just-respawned) node is
					// running again — lift any crash indicator.
					if (t.crashed) t.crashed = false;
					// The node carries its current error on the state plane (always on,
					// re-pushed): a lost first PROCESSING_ERROR still surfaces here, and a
					// healthy respawn's null clears the stale chip. Applied unconditionally
					// when present so backend truth wins even when no diff-driven `error`
					// event fired.
					if ('error' in ev.payload) t.error = ev.payload.error ?? null;
					// The node advertises its SSE log endpoint here; surfacing it lets
					// the Console subscribe peer-to-peer (see $lib/stores/logStream).
					if (ev.payload.log_endpoint !== undefined) t.log_endpoint = ev.payload.log_endpoint;
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
						// A terminal boot failure after a crash: lift the transient crash
						// indicator (which outranks 'error' in nodeHealth) so the traceback
						// shows instead of a 'restarting…' badge that never resolves.
						if (ev.payload.stage === 'error' && t.crashed) {
							t.crashed = false;
							t.crashExit = null;
						}
				}
				break;
			}
			case 'node_crashed': {
				// The node's OS process died; the manager is auto-restarting it. Show
				// a distinct, self-recovering crash state (cleared by the next healthy
				// state_update above), not a code error.
				const t = this.nodeById(ev.payload.node);
				if (t) {
					t.crashed = true;
					t.restarts = ev.payload.restarts;
					t.crashExit = ev.payload.exitcode;
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
				// Live snapshot — drives the node's red border, the floating error
				// chip, and the inspector's current-error section. Always on, via the
				// control plane, independent of whether a Console is open. The backend
				// also fires this per ANCESTOR INSTANCE of an errored member (keyed by the
				// instance uid) so a collapsed sub-patch reflects a deep member's error
				// live — update the mirrored `inst.error` field (which feeds the synth
				// node's border) rather than the throwaway synth node object.
				const inst = this.instances[ev.payload.node];
				if (inst) {
					inst.error = ev.payload.error;
				} else {
					const t = this.nodeById(ev.payload.node);
					if (t) t.error = ev.payload.error;
					// Console ingest only for real nodes — an ancestor-instance event just
					// mirrors a member error that already ingested under the member's uid.
					if (ev.payload.error)
						consoleStore().ingestError(ev.payload.node, ev.payload.error, Date.now());
				}
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
			case 'manager_shutdown':
				this.connected = false;
				break;
		}
	}

	private async _refreshNodeTypes(): Promise<void> {
		try {
			const result = await this.ctl.call<{ types: NodeTypeInfo[] }>('list_nodes');
			this.nodeTypes = result.types;
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

	async addNode(
		type: string,
		category: string,
		pos: [number, number],
		instId?: string
	): Promise<string> {
		// `instId` lands the node inside that sub-patch (member of the instance);
		// omitted, it goes in the root graph.
		const uid =
			(await this.ctl.call<string>('add_node', { type, category, pos, inst_id: instId })) ?? '';
		// Record AFTER the call so we know the backend-assigned uid + display name —
		// a redo restores both so links/panels reconnect to the same node.
		if (uid)
			this._record({
				kind: 'add_node',
				label: `Add ${type}`,
				domain: 'graph',
				context: captureNavContext(),
				payload: { type, category, pos, instId, uid, name: this.nodeById(uid)?.name }
			});
		return uid;
	}

	async removeNode(uid: string, captureLinks = true): Promise<void> {
		// Deleting a collapsed sub-patch INSTANCE tears down a whole subtree — members, internal
		// links, boundaries, and external wires (which reference inner-leaf uids, so they are not
		// even caught by a uid link filter). nodeById(instanceUid) is only a SYNTHETIC node
		// (type 'Sub-patch'), so a generic remove_node action would replay an uninstantiable
		// add_node{type:'Sub-patch'} on undo and lose the subtree forever. Record it as a
		// full-graph checkpoint instead — the proven load_patch machinery restores any subtree
		// (nested/shared/external wires) exactly, with no fragile hand-reconstruction.
		const inst = this.instances[uid];
		if (inst && uid !== ROOT_ID) {
			const record = !history().isSuspended;
			const before = record ? await this._loadCheckpointBefore() : null;
			await this.ctl.call('remove_node', { node: uid });
			if (record && before) {
				let afterYaml = '';
				try {
					afterYaml = (await this.serialize()).yaml;
				} catch {
					/* whole graph gone — empty after-state */
				}
				this._record({
					kind: 'load_patch',
					label: `Delete ${inst.name}`,
					domain: 'graph',
					context: captureNavContext(),
					payload: {
						beforeYaml: before.yaml,
						afterYaml,
						beforeLayout: before.layout as WorkspaceState | null,
						afterLayout: workspace().serialize() as WorkspaceState | null
					}
				});
			}
			return;
		}
		// Capture the full node + its links BEFORE the backend tears them down.
		// A batch delete (removeNodes) passes captureLinks=false and records the links
		// as separate remove_link actions ordered first, so undo restores every node
		// before any link (a link to a co-deleted node can't be re-added until both
		// endpoints are back).
		const node = this.nodeById(uid);
		if (node && !history().isSuspended) {
			const links = captureLinks
				? this.links.filter((l) => l.node_in === uid || l.node_out === uid).map((l) => ({ ...l }))
				: [];
			this._record({
				kind: 'remove_node',
				label: `Delete ${node.name}`,
				domain: 'graph',
				context: captureNavContext(),
				payload: {
					uid,
					node: structuredClone($state.snapshot(node)),
					links,
					membership: node.membership ?? null,
					boundPanels: workspace().panelsBoundTo(uid)
				}
			});
		}
		await this.ctl.call('remove_node', { node: uid });
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
		// The single-source rule may displace an existing wire on the input — capture
		// it before the add so undo can restore it. A MULTI input slot accepts many
		// wires (the backend appends, never evicts), so nothing is displaced there.
		const targetMulti = this._realNode(link.node_in)?.input_multi?.includes(link.slot_in) ?? false;
		const displaced = targetMulti
			? null
			: (this.links.find((l) => l.node_in === link.node_in && l.slot_in === link.slot_in) ?? null);
		this._record({
			kind: 'add_link',
			label: 'Connect',
			domain: 'graph',
			context: captureNavContext(),
			payload: { link: { ...link }, displaced: displaced ? { ...displaced } : null }
		});
		await this.ctl.call('add_link', link as unknown as Record<string, unknown>);
	}

	async removeLink(link: LinkInfo): Promise<void> {
		this._record({
			kind: 'remove_link',
			label: 'Disconnect',
			domain: 'graph',
			context: captureNavContext(),
			payload: { link: { ...link } }
		});
		await this.ctl.call('remove_link', link as unknown as Record<string, unknown>);
	}

	async updateParam(node: string, group: string, name: string, value: unknown): Promise<void> {
		// Key on the param's EXISTENCE, not its value's truthiness — a real param may hold
		// 0/false/''. A missing param (agent typo, or a call racing node hydration) would
		// otherwise record oldValue=undefined, and the undo's update_param drops `value`
		// from the JSON entirely → backend KeyError. Fail loudly instead of poisoning undo.
		const param = this.nodeById(node)?.params?.[group]?.[name];
		if (!param) throw new Error(`update_param: no param ${group}.${name} on node ${node}`);
		const oldValue = param.value;
		this._record({
			kind: 'update_param',
			label: `Set ${name}`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { node, group, name, oldValue, newValue: value }
		});
		this.writeParamValue(node, group, name, value);
	}

	/** Client leaf-write of a committed param value into the CRDT doc (Phase 3 — replaces the
	 * `update_param` RPC). The manager applies it to the graph via `apply_client_write` (re-
	 * projecting to shared siblings) and mirrors it back; the reader overlay reflects it locally
	 * and on the mirror-back. Records no history — callers (updateParam, the executor) own that. */
	writeParamValue(node: string, group: string, name: string, value: unknown): void {
		this._sync.commit((doc) => {
			setParamValue(doc, node, group, name, value as number | string | boolean);
		});
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
		// racing node hydration) would otherwise leaf-write an `expr` onto a phantom param entry the
		// graph rejects and the re-mirror never prunes — an orphan binding lingering in the doc.
		if (!d) throw new Error(`set_expression: no param ${group}.${name} on node ${node}`);
		const oldExpr: ExprState = {
			expression: d.expression ?? null,
			enabled: d.expression_enabled ?? false,
			triggers_process: d.expression_triggers_process ?? false
		};
		const newExpr: ExprState = {
			expression,
			enabled: opts.enabled ?? false,
			triggers_process: opts.triggers_process ?? false
		};
		this._record({
			kind: 'set_expression',
			label: `Set ${name} expression`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { node, group, name, oldExpr, newExpr }
		});
		this.writeExpression(node, group, name, newExpr);
	}

	/** Client leaf-write of a param's expression binding into the CRDT doc (Phase 3 — replaces the
	 * `set_expression` RPC). The manager applies it via `set_member_expression` (re-projecting to
	 * shared siblings) and echoes the runtime-enriched param descriptor as a `state_update`, so the
	 * fx toggle + `expression_error` indicator refresh exactly as before. Records no history —
	 * callers (setExpression, the executor) own that. A null/empty expression clears the binding. */
	writeExpression(node: string, group: string, name: string, expr: ExprState): void {
		this._sync.commit((doc) =>
			docSetParamExpr(
				doc,
				node,
				group,
				name,
				expr.expression
					? { source: expr.expression, enabled: expr.enabled, triggers: expr.triggers_process }
					: null
			)
		);
	}

	async setNodePos(uid: string, pos: [number, number]): Promise<void> {
		const oldPos = this.nodeById(uid)?.pos ?? [0, 0];
		this._record({
			kind: 'set_node_pos',
			label: `Move ${this.nodeById(uid)?.name ?? uid}`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { uid, oldPos: [oldPos[0], oldPos[1]], newPos: pos }
		});
		this.writeNodePos(uid, pos);
	}

	/** Client leaf-write of a committed node/instance position into the CRDT doc (Phase 3 —
	 * replaces the `set_node_pos` RPC). The manager applies it via `set_member_pos` (mirroring
	 * shared members to siblings) and re-mirrors; the doc-pos overlay in `_syncFromDoc` reflects
	 * it. Records no history — callers (setNodePos, the executor) own that. Committed on drag-stop
	 * only; live drag stays local to Svelte Flow (never the doc). */
	writeNodePos(uid: string, pos: [number, number]): void {
		this._sync.commit((doc) => docSetNodePos(doc, uid, pos));
	}

	/** Set a node's mutable display name (uid identity is unchanged). */
	async renameNode(uid: string, name: string): Promise<void> {
		const node = this.nodeById(uid);
		const oldName = node?.name ?? '';
		if (oldName === name) return;
		this._record({
			kind: 'rename_node',
			label: `Rename ${oldName} → ${name}`,
			domain: 'graph',
			context: captureNavContext(),
			payload: { uid, oldName, newName: name }
		});
		await this.ctl.call('rename_node', { node: uid, name });
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

	async loadText(content: string): Promise<void> {
		const before = await this._loadCheckpointBefore();
		await this.ctl.call('load_text', { content });
		this._recordLoadPatch(before, content);
	}

	/** Raw patch swap used by the load_patch undo executor — bypasses recording. */
	async applyLoadCheckpoint(yaml: string): Promise<void> {
		await this.ctl.call('load_text', { content: yaml });
	}

	/** Snapshot the current patch + layout before a destructive load. */
	private async _loadCheckpointBefore(): Promise<{ yaml: string; layout: unknown }> {
		let yaml = '';
		try {
			yaml = (await this.serialize()).yaml;
		} catch {
			/* nothing loaded yet — empty before-state */
		}
		return { yaml, layout: workspace().serialize() };
	}

	private _recordLoadPatch(before: { yaml: string; layout: unknown }, afterYaml: string): void {
		const payload = {
			beforeYaml: before.yaml,
			afterYaml,
			beforeLayout: before.layout as WorkspaceState | null,
			afterLayout: null as WorkspaceState | null
		};
		this._record({ kind: 'load_patch', label: 'Load patch', domain: 'graph', context: captureNavContext(), payload });
		// The post-load layout hasn't settled yet (the graph_replaced echo hydrates
		// it next); capture it onto this payload when it lands so redo is exact.
		this._pendingLoadAfter = payload;
	}

	/** Fill a pending load_patch's `afterLayout` with the now-settled arrangement.
	 * Called once the wholesale-load echo has hydrated the loaded patch's layout. */
	private _captureLoadAfterLayout(): void {
		if (!this._pendingLoadAfter) return;
		this._pendingLoadAfter.afterLayout = workspace().serialize();
		this._pendingLoadAfter = null;
	}

	/** Group the named nodes into a unique (inline) sub-patch. Returns its instance id. */
	async groupNodes(members: string[], pos?: [number, number]): Promise<string> {
		const r = await this.ctl.call<{ inst_id: string }>('group_nodes', { members, pos });
		if (r?.inst_id)
			this._record({
				kind: 'group_nodes',
				label: 'Group nodes',
				domain: 'graph',
				context: captureNavContext(),
				payload: { members: [...members], instId: r.inst_id, pos }
			});
		return r.inst_id;
	}

	/** Dissolve a sub-patch instance back into its member nodes. */
	async expandInstance(instId: string): Promise<void> {
		const iface = structuredClone($state.snapshot(this.instances[instId]?.interface ?? {}));
		const r = await this.ctl.call<{ restored: string[] }>('expand_instance', { inst_id: instId });
		this._record({
			kind: 'expand_instance',
			label: 'Ungroup',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, restoredMembers: r?.restored ?? [], interface: iface }
		});
	}

	/** Promote an instance to shared and spawn a strict-mirror sibling copy. */
	async duplicateShared(instId: string, pos?: [number, number]): Promise<void> {
		const wasUnique = !this.instances[instId]?.def_id;
		const r = await this.ctl.call<{ inst_id: string }>('duplicate_shared', { inst_id: instId, pos });
		this._record({
			kind: 'duplicate_shared',
			label: 'Duplicate shared',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, newInstId: r?.inst_id ?? '', wasUnique, pos }
		});
	}

	/** Detach a shared instance into its own private (unique) copy. */
	async makeUnique(instId: string): Promise<void> {
		const defIdBefore = this.instances[instId]?.def_id ?? null;
		this._record({
			kind: 'make_unique',
			label: 'Make unique',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, defIdBefore }
		});
		await this.ctl.call('make_unique', { inst_id: instId });
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
		if (r?.bnd_id)
			this._record({
				kind: 'add_boundary',
				label: 'Add boundary',
				domain: 'graph',
				context: captureNavContext(),
				payload: { instId, bndId: r.bnd_id, dir, dtype, pos }
			});
		return r.bnd_id;
	}

	/** Set (inner_node/slot) or clear (nulls) a boundary's single inner target. */
	async wireBoundary(
		instId: string,
		bndId: string,
		innerNode: string | null,
		innerSlot: string | null
	): Promise<void> {
		const prev = this.instances[instId]?.interface?.[bndId];
		this._record({
			kind: 'wire_boundary',
			label: 'Wire boundary',
			domain: 'graph',
			context: captureNavContext(),
			payload: {
				instId,
				bndId,
				oldInner: { node: prev?.inner_node ?? null, slot: prev?.inner_slot ?? null },
				newInner: { node: innerNode, slot: innerSlot }
			}
		});
		await this.ctl.call('wire_boundary', {
			inst_id: instId,
			bnd_id: bndId,
			inner_node: innerNode,
			inner_slot: innerSlot
		});
	}

	/** Delete an In/Out boundary node (tears down its external wires). */
	async removeBoundary(instId: string, bndId: string): Promise<void> {
		const port = this.instances[instId]?.interface?.[bndId];
		if (port)
			this._record({
				kind: 'remove_boundary',
				label: 'Remove boundary',
				domain: 'graph',
				context: captureNavContext(),
				payload: { instId, bndId, port: structuredClone($state.snapshot(port)) }
			});
		await this.ctl.call('remove_boundary', { inst_id: instId, bnd_id: bndId });
	}

	/** Rename an In/Out portal (its label + the sub-patch's exposed slot name). The
	 * routing key (bndId) is unchanged, so external wires survive. Records undo only
	 * AFTER the RPC lands, so a rejected rename (blank/duplicate) doesn't poison history. */
	async renameBoundary(instId: string, bndId: string, name: string): Promise<void> {
		const oldName = this.instances[instId]?.interface?.[bndId]?.name ?? bndId;
		if (name === oldName) return;
		await this.ctl.call('rename_boundary', { inst_id: instId, bnd_id: bndId, name });
		this._record({
			kind: 'rename_boundary',
			label: 'Rename boundary',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, bndId, oldName, newName: name }
		});
	}

	/** Move an In/Out pill inside the entered view (mirrors across shared siblings). */
	async setBoundaryPos(instId: string, bndId: string, pos: [number, number]): Promise<void> {
		const old = this.instances[instId]?.interface?.[bndId]?.pos ?? [0, 0];
		this._record({
			kind: 'set_boundary_pos',
			label: 'Move boundary',
			domain: 'graph',
			context: captureNavContext(),
			payload: { instId, bndId, oldPos: [old[0], old[1]], newPos: pos }
		});
		await this.ctl.call('set_boundary_pos', { inst_id: instId, bnd_id: bndId, pos });
	}

	/** List one directory level on the BACKEND filesystem (full FS, no jail). */
	async listDir(path?: string): Promise<DirListing> {
		return this.ctl.call<DirListing>('list_dir', { path });
	}

	/** The bundled example patches (empty under a wheel without examples/). */
	async listExamples(): Promise<{ entries: FsEntry[] }> {
		return this.ctl.call<{ entries: FsEntry[] }>('list_examples');
	}

	/** Load a patch from a BACKEND filesystem path (destructive — replaces the graph). */
	async load(path: string): Promise<void> {
		const before = await this._loadCheckpointBefore();
		await this.ctl.call('load', { path });
		// The loaded patch's YAML, for redo (re-applying the same load).
		let afterYaml = '';
		try {
			afterYaml = (await this.serialize()).yaml;
		} catch {
			/* ignore */
		}
		this._recordLoadPatch(before, afterYaml);
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

	/** The uid of a member given its template-local name within `instId` — a direct
	 * read now that members is keyed local -> {uid, is_instance}. */
	memberUid(instId: string, local: string): string | null {
		return this.instances[instId]?.members[local]?.uid ?? null;
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
			consoleStore().forgetNodeDedup(old.uid);
			workspace().clearNodeRefs(old.uid);
		}
		this.nodes = next.map((n) => {
			const cur = byUid.get(n.uid);
			if (cur) {
				// subpatch_changed is a STRUCTURE event (group/expand/share/make-unique):
				// membership/names/instances moved but the live node processes are
				// unchanged. Their runtime lifecycle state (stage/error/stats/restarts/
				// log_endpoint) is owned by the state_update / node_* stream, and this
				// snapshot was built on a manager thread a later state_update may have
				// overtaken — so copying its volatile fields would REGRESS them (a ready
				// node flickering back to a boot spinner, a cleared error chip reappearing,
				// live stats blanked). Refresh the structural fields in place; keep the
				// survivor's runtime state, which its authoritative stream owns.
				Object.assign(cur, n, {
					stage: cur.stage,
					error: cur.error,
					stats: cur.stats,
					restarts: cur.restarts,
					log_endpoint: cur.log_endpoint
				});
				return cur;
			}
			this._seedNodeViewerState(n); // genuinely new node — seed its inline view state
			return n;
		});
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

	/** Fold an entity into its owning scope's members map.
	 * Root ≡ a scope: a top-level node carries membership {instance: ROOT_ID, …}, a member
	 * carries {instance: <sub-patch>, …}. The map (local -> {uid, is_instance}) is the index
	 * the editor renders a scope's direct children from, so an incremental add/remove must
	 * update it exactly as a subpatch_changed snapshot would. A node is never a nested
	 * instance, so is_instance is always false here. */
	private _addScopeMember(
		membership: { instance: string; local_name: string } | null,
		uid: string
	): void {
		if (!membership) return;
		const inst = this.instances[membership.instance];
		// An instance always exists before its members' events (it's created via
		// group_nodes/subpatch_changed first), so a miss is a real backend↔store desync,
		// not a normal case — surface it instead of silently orphaning the node.
		if (!inst) {
			console.warn(`_addScopeMember: unknown scope ${membership.instance} for ${uid}`);
			return;
		}
		inst.members[membership.local_name] = { uid, is_instance: false };
	}

	/** Drop an entity from its owning scope's members map (mirror of _addScopeMember). */
	private _removeScopeMember(membership: { instance: string; local_name: string } | null): void {
		if (!membership) return;
		const inst = this.instances[membership.instance];
		if (!inst) {
			console.warn(`_removeScopeMember: unknown scope ${membership.instance}`);
			return;
		}
		delete inst.members[membership.local_name];
	}

	/** Build the virtual NodeInstanceInfo that stands in for a sub-patch instance
	 * wherever a node is looked up by name. A sub-patch behaves exactly like a
	 * node: its WIRED boundaries become real input/output slots (so the canvas
	 * renders connectors + output viewers and seeds add-node clicks the same way
	 * as any node — the /data route splices an output boundary to its inner
	 * member's stream). Live sharing state is recomputed by the inspector from
	 * `instances`; the marker just rides the id + glyph/badge bits. */
	private _synthSubpatchNode(instId: string, inst: InstanceInfo): NodeInstanceInfo {
		const shared = Boolean(inst.def_id);
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
		const sig = `${inst.name}|${inst.kind}|${error ?? ''}|${memberCount}|${JSON.stringify(inst.slots)}|${labelSig}`;

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
			type: shared ? 'Shared sub-patch' : 'Sub-patch',
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
			subpatch: { instId, shared, memberCount }
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
				// the instance); omitted, they go in the root graph.
				const newUid = await this.addNode(s.type, s.category, s.pos, instId);
				rename[s.key] = newUid;
				for (const [group, params] of Object.entries(s.params)) {
					for (const [name, value] of Object.entries(params)) {
						try {
							await this.updateParam(newUid, group, name, value);
						} catch {
							/* ignore a single rejected param */
						}
					}
				}
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
		const uidSet = new Set(uidList);
		const label = `Delete ${uidList.length} node${uidList.length > 1 ? 's' : ''}`;

		// A collapsed sub-patch instance in the batch forces the WHOLE batch to be ONE full-graph
		// checkpoint. Deleting an instance is undone by reloading the patch, and load_doc re-mints
		// every uid on reload — which cannot compose with the incremental, uid-referencing
		// link/node inverses of a compound transaction: a trailing add_link inverse would target a
		// re-minted survivor's now-dead uid, abort the replay mid-way, and half-revert the graph.
		// Capture one before/after checkpoint and suspend the per-child records; the backend
		// cascades link cleanup on node/instance removal, so no explicit remove_link is needed
		// (undo restores every link from the checkpoint YAML).
		if (uidList.some((u) => this.instances[u] && u !== ROOT_ID) && !history().isSuspended) {
			const before = await this._loadCheckpointBefore();
			await history().suspend(async () => {
				for (const uid of uidList) await this.removeNode(uid, false);
			});
			let afterYaml = '';
			try {
				afterYaml = (await this.serialize()).yaml;
			} catch {
				/* whole graph gone — empty after-state */
			}
			this._record({
				kind: 'load_patch',
				label,
				domain: 'graph',
				context: captureNavContext(),
				payload: {
					beforeYaml: before.yaml,
					afterYaml,
					beforeLayout: before.layout as WorkspaceState | null,
					afterLayout: workspace().serialize() as WorkspaceState | null
				}
			});
			return;
		}

		// Snapshot the affected links up front (one read — not re-derived between the
		// awaits below, where the store may not have caught up to each removal yet).
		const links = this.links
			.filter((l) => uidSet.has(l.node_in) || uidSet.has(l.node_out))
			.map((l) => ({ ...l }));
		await history().transaction(label, async () => {
			for (const link of links) await this.removeLink(link);
			for (const uid of uidList) await this.removeNode(uid, false);
		});
	}

}

let _store: GraphStore | null = null;
export function graph(): GraphStore {
	if (!_store) _store = new GraphStore();
	return _store;
}
