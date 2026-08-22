/** Central reactive graph state, backed by the control WS. The store owns the only writes, so a
 * component just reads its `$state` fields. */
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
	type NodeTypeInfo,
	type ScanDiff
} from '$lib/api/control';
import { boundaryType } from '$lib/api/vocab';
import { consoleStore } from './console.svelte';
import { selection } from './selection.svelte';
import { workspace } from 'panelty';
import type { SlotView } from '$lib/viewers/inlineView';
import { history, type Action } from './history.svelte';
import { captureNavContext } from '$lib/stores/navContext';
import { ROOT_ID } from '$lib/editor/subpatchScene';
import { SyncClient } from '$lib/crdt/syncClient';
import {
	linkViews,
	nodeViews,
	instanceViews,
	docParams,
	viewersJson,
	globalViews,
	arrangementTabs,
	type Doc,
	type GlobalView,
	type GlobalType
} from '$lib/crdt/graphDoc';
import { assembleNode, type RuntimeOverlay } from '$lib/crdt/nodeAssembly';
import { assembleInstances, instanceError } from '$lib/crdt/instanceAssembly';
import type { StringParam } from '$lib/api/types';

/** Safety net: lift a ⟳ spinner after this long when a node never reports the refresh done.
 * Generous — an LSL resolve blocks the node's ctrl thread ~4s. */
const REFRESH_SPINNER_TIMEOUT_MS = 15000;

/** Stable key for an in-flight param refresh; U+001F cannot occur in a uid/group/name. */
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
	/** Latches on the first connect and never clears — see {@link disconnected}. */
	private _everConnected = $state(false);
	hadHello = $state(false);

	/** Patch globals (system + user), doc-authoritative, in system-first/creation order. */
	globals = $state<GlobalView[]>([]);

	/** Bumps on every WHOLESALE graph load, never on an incremental add/remove; editors re-fit on it. */
	loadEpoch = $state(0);

	nodeTypes = $state<NodeTypeInfo[] | null>(null);

	/** Params with a ⟳ refresh in flight → its safety-timeout handle. Cleared when the node reports
	 * the param done (`refreshed_params`), never on the fire-and-forget RPC ack. */
	private _refreshing = $state<Record<string, ReturnType<typeof setTimeout>>>({});

	/** instance_id of the manager we last hydrated from; a change is a fresh session, not a reconnect. */
	private _lastInstanceId: string | null = null;

	/** The last snapshot's per-node runtime overlay, consumed by `_seedRuntime`. */
	private _snapshotRuntime: GraphSnapshot['runtime'] = {};

	/** The control client (injectable for tests; defaults to the live WS one). */
	private ctl: Control;

	/** The document driver — the browser replica of the manager's control-plane document. */
	private _sync: SyncClient;

	/** The connection was ESTABLISHED and is now gone. Not `!connected`: at boot that would alarm
	 * for the few hundred ms before the socket first opens. */
	get disconnected(): boolean {
		return this._everConnected && !this.connected;
	}

	constructor(ctl: Control = getControl()) {
		this.ctl = ctl;
		ctl.onConnect((c) => {
			this.connected = c;
			if (c) this._everConnected = true;
		});
		ctl.on((ev) => this._handle(ev));
		this._sync = new SyncClient(ctl);
		this._sync.onDocChange(() => this._syncFromDoc());
		this._sync.start();
	}

	/** The control-plane document — the one projection a client reads. */
	get doc(): Doc {
		return this._sync.doc;
	}

	/** Whether the replica has pulled from the manager yet; until then doc reads describe nothing. */
	get docSynced(): boolean {
		return this._sync.synced;
	}

	/** Re-derive every document-owned subtree, after each applied change. Runtime state
	 * (error/stage/ufreq) and catalog metadata (slots/category) stay event-sourced. */
	private _syncFromDoc(): void {
		const doc = this._sync.doc;
		this.links = linkViews(doc);
		this.globals = globalViews(doc);
		// The workspace store rebuilds its tree from this; the client holds no second copy.
		workspace().syncFromDoc(arrangementTabs(doc));
		// Both reconcilers no-op until the catalog lands, then rebuild from the doc.
		this._reconcileNodesFromDoc();
		this._reconcileInstancesFromDoc();
	}

	/** Apply a wholesale snapshot, returning whether it came from a NEW backend session — which is
	 * what a same-session reconnect must not look like. */
	private _replaceSnapshot(snap: GraphSnapshot, wholesale: boolean): boolean {
		// Absent → an older backend; keep whatever the async fetch set.
		if (snap.node_types?.length) this.nodeTypes = snap.node_types;
		// The snapshot and the doc delta ride separate channels in no defined order, so the runtime
		// overlay is both stashed for nodes still to materialize and applied to those already here.
		this._snapshotRuntime = snap.runtime ?? {};
		for (const [uid, rt] of Object.entries(this._snapshotRuntime)) {
			const node = this._realNode(uid);
			if (!node) continue;
			node.stage = rt.stage;
			node.error = rt.error ?? null;
		}
		// A same-session reconnect fires no doc transaction, so nothing else refreshes the facades.
		this._recomputeInstanceErrors();
		this.savePath = snap.save_path;
		this.unsavedChanges = snap.unsaved_changes;

		// The arrangement rides the doc; what the snapshot carries is the VIEWPOINT, this client's
		// alone — persisted, never converged.
		const freshSession = snap.instance_id !== this._lastInstanceId;
		this._lastInstanceId = snap.instance_id;
		if (snap.viewpoint != null) workspace().restoreViewpoint(snap.viewpoint);
		return freshSession;
	}

	/** Drop every projection assembled from the OUTGOING document. Reconciliation runs only from the
	 * doc observer, and a fresh manager with an EMPTY graph answers our SV with no change at all. */
	private _resetProjection(): void {
		this.nodes = [];
		this.links = [];
		this.instances = {};
		this.globals = [];
		// `_snapshotRuntime` is NOT cleared: `_replaceSnapshot` ran first, so it already holds the
		// INCOMING session's overlay. The arrangement store is a separate singleton, so it is here.
		workspace().syncFromDoc([]);
	}

	private _onWholesaleLoad(): void {
		this.loadEpoch += 1;
		// A wholesale load mints new uids and clears the manager's history, so a kept client entry
		// would pop against a command that is not there. A same-session reconnect never comes here.
		history().reset();
		consoleStore().clear();
		selection().forgetAll();
	}

	/** Write ONE slot's inline view, merging into the node's blob. The kind stored is the user's RAW
	 * pick; a sub-patch SCOPE has no engine blob, so its record holds it client-side. */
	setSlotView(uid: string, slot: string, view: SlotView): void {
		const node = this.nodeById(uid);
		if (!node?.output_slots[slot]) return;
		const inst = this.instances[uid];
		if (inst) {
			// From the node's CURRENT output slots, never the blob's keys: a blob saved before a
			// node file changed its slots carries a name that is no longer a slot.
			const viewers: NodeInstanceInfo['viewers'] = {};
			for (const s of Object.keys(node.output_slots)) {
				const stored = node.viewers?.[s];
				if (s === slot) viewers[s] = { ...stored, ...view };
				else if (stored) viewers[s] = { ...stored };
			}
			inst.viewers = viewers;
			return;
		}
		void this.ctl
			.call('edit_node', { node: uid, viewers: { [slot]: view } })
			.then(() => this._recordGraphCmd(`Set ${slot} view`))
			.catch(() => {
				/* soft view state — the next edit re-sends it */
			});
	}

	private _handle(ev: ControlEvent): void {
		switch (ev.event) {
			case 'hello': {
				// Not wholesale: a `hello` is also what a transient reconnect delivers.
				const fresh = this._replaceSnapshot(ev.payload, false);
				this.hadHello = true;
				if (fresh) {
					// A NEW session mints uids from 1 again, so the stale replica must fall NOW,
					// synchronously, before this connection answers the server's binary hello SV.
					// Projections first: `_resetProjection` reads `this.nodes`.
					this._resetProjection();
					this._sync.reset();
					this._onWholesaleLoad();
				}
				// The catalog usually rides on the hello snapshot; fetch it only if one omitted it.
				if (!this.nodeTypes?.length) void this._refreshNodeTypes();
				break;
			}
			case 'graph_replaced':
				this._replaceSnapshot(ev.payload, true);
				this._onWholesaleLoad();
				break;
			case 'state_update': {
				const t = this.nodeById(ev.payload.node);
				if (t) {
					// Params are doc-owned: merge ONLY the runtime bits, never wholesale-replace, which
					// would clobber the reconcile's value+descriptor assembly.
					this._mergeParamRuntime(t, ev.payload.params);
					if (ev.payload.stage) t.stage = ev.payload.stage;
					// The state plane re-pushes the current error, so backend truth wins here even
					// when no diff-driven `error` event fired.
					if ('error' in ev.payload) {
						t.error = ev.payload.error ?? null;
						// A collapsed sub-patch's health is derived and cached, so every writer of a
						// node's `error` has to invalidate it.
						this._recomputeInstanceErrors();
					}
				}
				// Lift each spinner exactly when the fresh options land. Keyed by node, not by `t`.
				for (const [group, name] of ev.payload.refreshed_params ?? []) {
					this._endRefresh(refreshKey(ev.payload.node, group, name));
				}
				break;
			}
			case 'node_stage': {
				// Discrete stage transitions the state plane cannot carry — today only a bootstrap error.
				const t = this._realNode(ev.payload.node);
				if (t) {
					t.stage = ev.payload.stage;
					if (ev.payload.error !== undefined) {
						t.error = ev.payload.error;
						this._recomputeInstanceErrors(); // derived facade health — see `state_update`
					}
				}
				break;
			}
			case 'node_stats': {
				const t = this.nodeById(ev.payload.node);
				if (t) t.stats = ev.payload.stats;
				break;
			}
			case 'param_values': {
				// Applied surgically to the existing descriptors, so a re-evaluation preview cannot
				// clobber a concurrent edit on a sibling param.
				const t = this.nodeById(ev.payload.node);
				if (t) {
					for (const [group, names] of Object.entries(ev.payload.values)) {
						for (const [name, value] of Object.entries(names)) {
							const p = t.params[group]?.[name];
							// Widen past the union's narrowed `value`; the backend guarantees the type.
							if (p) (p as { value: unknown }).value = value;
						}
					}
				}
				break;
			}
			case 'error': {
				// Always keyed by a REAL node uid; a sub-patch's deep error is derived, and this event
				// fires no doc transaction, so the enclosing instances are recomputed by hand.
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
			case 'node_types':
				this._applyNodeTypes(ev.payload.types);
				break;
		}
	}

	private async _refreshNodeTypes(): Promise<void> {
		try {
			const result = await this.ctl.call<{ types: NodeTypeInfo[] }>('list_nodes');
			this._applyNodeTypes(result.types);
		} catch (e) {
			console.warn('list_nodes failed', e);
		}
	}

	/** Adopt a palette catalog. It supplies the descriptors, so nodes and instances rebuild here. */
	private _applyNodeTypes(types: NodeTypeInfo[]): void {
		this.nodeTypes = types;
		this._reconcileNodesFromDoc();
		this._reconcileInstancesFromDoc();
	}

	/** Re-derive the node registry from disk and report what changed; explicit, since there is no
	 * watcher. The fresh catalog arrives as a `node_types` event in every open tab. */
	async rescanNodes(): Promise<ScanDiff> {
		return this.ctl.call<ScanDiff>('rescan_nodes', {});
	}

	/** Where this patch's workspace files live — a per-run temp directory under a random name. It
	 * rides `get_patch` beside the save path, because both answer "where does this patch live". */
	async openWorkspace(): Promise<string> {
		const r = await this.ctl.call<{ workspace: string }>('get_patch', {});
		return r.workspace;
	}

	/** Push an action onto the history (unless a replay is in progress). */
	private _record(action: Action): void {
		if (!history().isSuspended) history().record(action);
	}

	/** Record ONE graph command. The manager owns the exact inverse, so this only marks the step. */
	private _recordGraphCmd(label: string): void {
		this._record({ kind: 'graph_cmd', domain: 'graph', label, context: captureNavContext() });
	}

	async addNode(
		type: string,
		category: string,
		pos: [number, number],
		instId?: string,
		params?: Record<string, Record<string, unknown>>
	): Promise<string> {
		// `params` are applied at creation UNDER THE GRAPH LOCK: a post-add leaf write would no-op
		// until the new node syncs into the replica, silently dropping the values.
		const born = await this.ctl.call<{ uid: string }>('add_node', {
			type,
			category,
			pos,
			inst_id: instId,
			params
		});
		const uid = born?.uid ?? '';
		if (uid) this._recordGraphCmd(`Add ${type}`);
		return uid;
	}

	async removeNode(uid: string): Promise<void> {
		// The label reads the node's name before it vanishes.
		const label = `Delete ${this.nodeById(uid)?.name ?? uid}`;
		await this.ctl.call('remove_node', { node: uid });
		this._recordGraphCmd(label);
	}

	/** Respawn a node in place, keeping its uid, name, params, position, membership and links. A
	 * recovery action rather than an edit, so it records no history. */
	async restartNode(uid: string): Promise<void> {
		await this.ctl.call('restart_node', { node: uid });
	}

	async addLink(link: LinkInfo): Promise<void> {
		await this.ctl.call('add_link', link as unknown as Record<string, unknown>);
		this._recordGraphCmd('Connect');
	}

	async removeLink(link: LinkInfo): Promise<void> {
		await this.ctl.call('remove_link', link as unknown as Record<string, unknown>);
		this._recordGraphCmd('Disconnect');
	}

	async updateParam(node: string, group: string, name: string, value: unknown): Promise<void> {
		// Guard on EXISTENCE, not truthiness — a real param may hold 0, false or ''.
		const param = this.nodeById(node)?.params?.[group]?.[name];
		if (!param) throw new Error(`edit_node: no param ${group}.${name} on node ${node}`);
		await this.ctl.call('edit_node', { node, params: { [group]: { [name]: value } } });
		this._recordGraphCmd(`Set ${name}`);
	}

	/** Add a NEW user global. The name is guarded HERE — `set_global` overwrites by design. */
	async addGlobal(name: string, value: number | string | boolean, type: GlobalType): Promise<void> {
		if (this.globals.some((g) => g.name === name)) throw new Error(`global ${name} already exists`);
		await this.ctl.call('set_global', { name, value, type });
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
		await this.ctl.call('set_global', { name });
		this._recordGraphCmd(`Remove global ${name}`);
	}

	/** Rename a user global; refs are NOT rewritten, so a stale `globals.<old>` throws at eval time.
	 * A set of the new name compounded with a delete of the old, so it is one undo step. */
	async renameGlobal(oldName: string, newName: string): Promise<void> {
		const held = this.globals.find((g) => g.name === oldName);
		if (!held) throw new Error(`no global ${oldName}`);
		await this.ctl.call('compound', {
			ops: [
				{ op: 'set_global', payload: { name: newName, value: held.value, type: held.type } },
				{ op: 'set_global', payload: { name: oldName } }
			]
		});
		this._recordGraphCmd(`Rename global ${oldName} → ${newName}`);
	}

	/** Ask a live node to re-evaluate a param's options. Options only, never the value, so it is
	 * not an undoable edit; the fresh list arrives on the node's next state_update. */
	async refreshParam(node: string, group: string, name: string): Promise<void> {
		const key = refreshKey(node, group, name);
		this._beginRefresh(key);
		try {
			await this.ctl.call('refresh_param', { node, group, name });
		} catch (e) {
			// A failed dispatch means the node never re-scans, so do not wait out the safety timeout.
			this._endRefresh(key);
			throw e;
		}
	}

	/** Whether a ⟳ refresh is in flight for this param. */
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
		if (!d) throw new Error(`edit_node: no param ${group}.${name} on node ${node}`);
		// `expression` is sent even when null: its PRESENCE is what clears a binding.
		await this.ctl.call('edit_node', {
			node,
			params: {
				[group]: {
					[name]: {
						expression: expression ?? '',
						mode: opts.enabled ? 'expression' : 'constant',
						triggers: opts.triggers_process ?? false
					}
				}
			}
		});
		this._recordGraphCmd(`Set ${name} expression`);
	}

	async setNodePos(uid: string, pos: [number, number]): Promise<void> {
		// Committed on drag-stop only; a live drag stays local to Svelte Flow.
		await this.ctl.call('edit_node', { node: uid, pos });
		this._recordGraphCmd(`Move ${this.nodeById(uid)?.name ?? uid}`);
	}

	/** Set a node's mutable display name (uid identity is unchanged). */
	async renameNode(uid: string, name: string): Promise<void> {
		const oldName = this.nodeById(uid)?.name ?? '';
		if (oldName === name) return;
		await this.ctl.call('edit_node', { node: uid, name });
		this._recordGraphCmd(`Rename ${oldName} → ${name}`);
	}

	/** Store where THIS client is looking. Persisted in the `.gfi`, but never converged and never
	 * dirtying: persistence and dirtiness are separate axes. */
	async setViewpoint(viewpoint: unknown): Promise<void> {
		try {
			await this.ctl.call('set_viewpoint', { viewpoint });
		} catch {
			/* not connected / in flight — ignore */
		}
	}

	/** Write the patch. Where it landed comes back from the MANAGER (`save_path_changed`), never
	 * latched from this reply — a latch names the patch only in the tab that saved it. */
	async save(path: string): Promise<{ path: string }> {
		// `path` is the whole payload and is REQUIRED; the arrangement is the manager's already.
		return this.ctl.call<{ path: string }>('save', { path });
	}

	/** Reset to an empty, unnamed patch. Nothing is written here: a New emits no
	 * `save_path_changed`, so the `graph_replaced` snapshot is the sole carrier of the null path. */
	async newPatch(): Promise<void> {
		await this.ctl.call('new', {});
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

	/** The sub-patch a boundary port belongs to, or null when the uid names no port. A port is not a
	 * leaf node, so `nodeById` does not answer for one; everything else about it is a node op. */
	portScope(uid: string): string | null {
		for (const [scope, inst] of Object.entries(this.instances)) {
			if (inst.interface[uid]) return scope;
		}
		return null;
	}

	/** List one directory level on the BACKEND filesystem (full FS, no jail). */
	async listDir(path?: string): Promise<DirListing> {
		return this.ctl.call<DirListing>('list_dir', { path });
	}

	/** Load a patch from a BACKEND filesystem path; destructive, and it resets the session, so
	 * there is no history entry. A `.gfi` is a zip, so a path is the only door the client has. */
	async load(path: string): Promise<void> {
		await this.ctl.call('load', { path });
	}

	/** A real node by uid (no sub-patch synthesis), or null. */
	private _realNode(uid: string): NodeInstanceInfo | null {
		return this.nodes.find((n) => n.uid === uid) ?? null;
	}

	/** Resolve a node by UID — the one accessor. A sub-patch instance id resolves to a VIRTUAL node
	 * whose own uid is the instance id, so selection, inspector and drag treat it like a node. */
	nodeById(id: string): NodeInstanceInfo | null {
		const real = this._realNode(id);
		if (real) return real;
		// ROOT is a real scope in the mirror, but it is the canvas — never a selectable node.
		if (id === ROOT_ID) return null;
		const inst = this.instances[id];
		if (!inst) {
			this._synthCache.delete(id);
			return null;
		}
		return this._synthSubpatchNode(id, inst);
	}

	/** Memoized virtual sub-patch nodes: a fresh object per call re-subscribed the inline viewer. */
	private _synthCache = new Map<string, { sig: string; node: NodeInstanceInfo }>();

	/** Validate that `uid` is a direct member of `instId` and return it. */
	memberUid(instId: string, uid: string): string | null {
		return this.instances[instId]?.members[uid]?.uid ?? null;
	}

	/** Reconcile the flat node list IN PLACE by uid, so a survivor keeps its object reference and
	 * with it its inline-viewer subscription. */
	private _reconcileNodes(next: NodeInstanceInfo[]): void {
		const byUid = new Map(this.nodes.map((n) => [n.uid, n]));
		this.nodes = next.map((n) => {
			const cur = byUid.get(n.uid);
			if (!cur) return n;
			Object.assign(cur, n);
			return cur;
		});
	}

	/** The runtime overlay for a node materializing from the doc for the FIRST time. A node created
	 * after the snapshot has no entry, and `creating` is what that means. */
	private _seedRuntime(uid: string): RuntimeOverlay {
		const seed = this._snapshotRuntime[uid];
		return { stage: seed?.stage ?? 'creating', error: seed?.error ?? null };
	}

	/** Pull the RUNTIME (event-sourced, never-in-the-doc) fields off a node so a re-assemble keeps them. */
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
				// An expression param DISPLAYS its live evaluated value, which never reaches the doc.
				if (p.expression_enabled) pr.liveValue = p.value;
				params[group][name] = pr;
			}
		}
		// `membership` is NOT extracted: the caller always re-derives it from the doc's forest.
		return {
			error: node.error,
			stage: node.stage,
			stats: node.stats,
			params
		};
	}

	/** Merge ONLY the runtime param bits from a state_update's descriptor map onto an existing node. */
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

	/** Derive a node's sub-patch membership from the doc's mirrored scope forest; ROOT → null. */
	private _membershipFromDoc(
		uid: string,
		index: Map<string, string>
	): { instance: string; local_name: string } | null {
		const instance = index.get(uid);
		return instance ? { instance, local_name: uid } : null;
	}

	/** uid → owning instance id, built ONCE per reconcile: per node it is an O(nodes × instances) walk. */
	private _membershipIndex(): Map<string, string> {
		const index = new Map<string, string>();
		for (const iv of instanceViews(this._sync.doc)) {
			for (const member of Object.keys(iv.members)) index.set(member, iv.uid);
		}
		return index;
	}

	/** Build `this.nodes` from the doc: each node is the doc's own fields, plus the catalog
	 * descriptor for its type, plus the runtime overlay the doc never holds. */
	private _reconcileNodesFromDoc(): void {
		if (!this.nodeTypes?.length) return; // no catalog yet → keep the current nodes; rebuild when it lands
		const doc = this._sync.doc;
		// Both indexes are built ONCE per reconcile rather than per node.
		const byType = new Map(this.nodeTypes.map((t) => [t.type, t]));
		const membership = this._membershipIndex();
		const next: NodeInstanceInfo[] = nodeViews(doc).map((nv) => {
			const existing = this._realNode(nv.uid);
			const catalog = byType.get(nv.type);
			const runtime: RuntimeOverlay = existing ? this._extractRuntime(existing) : this._seedRuntime(nv.uid);
			// A boundary port has no thread, so no `node_stage` will ever arrive for it — seeded
			// `creating` it would sit booting for the life of the patch.
			if (boundaryType(nv.type)) runtime.stage = 'ready';
			runtime.membership = this._membershipFromDoc(nv.uid, membership);
			const viewers = (viewersJson(doc, nv.uid) ?? {}) as NodeInstanceInfo['viewers'];
			return assembleNode(nv, docParams(doc, nv.uid), viewers, catalog, runtime);
		});
		this._reconcileNodes(next);
	}

	/** Build `this.instances` — the whole sub-patch forest, ROOT included — from the doc; `error` is
	 * DERIVED from the members' runtime errors, since the bridge never keys one by an instance uid. */
	private _reconcileInstancesFromDoc(): void {
		if (!this.nodeTypes?.length) return; // no catalog yet → keep event-sourced instances
		const doc = this._sync.doc;
		const nodes = nodeViews(doc).map((n) => ({ uid: n.uid, name: n.name }));
		const next = assembleInstances(instanceViews(doc), nodes, (uid) => this._realNode(uid)?.error ?? null);
		this._reconcileInstances(next);
	}

	/** Re-derive every instance's deep error in place: a member `error` event fires no doc
	 * transaction, so the reconcile that normally derives it does not run. */
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

	/** Reconcile the instances map IN PLACE by uid, so a survivor keeps its object reference. */
	private _reconcileInstances(next: Record<string, InstanceInfo>): void {
		for (const [uid, rec] of Object.entries(next)) {
			const cur = this.instances[uid];
			// A scope's `viewers` is not in the document, so the record IS its holder — the assembled
			// one is always empty and would blank a survivor's live view state.
			if (cur) Object.assign(cur, rec, { viewers: cur.viewers });
			else this.instances[uid] = rec;
		}
		for (const uid of Object.keys(this.instances)) {
			if (!(uid in next)) {
				delete this.instances[uid];
				this._synthCache.delete(uid); // cache lifetime tracks the instances map
			}
		}
	}

	/** Build the virtual NodeInstanceInfo that stands in for a sub-patch instance: its WIRED
	 * boundaries become real slots, so the canvas treats it exactly like a node. */
	private _synthSubpatchNode(instId: string, inst: InstanceInfo): NodeInstanceInfo {
		const error = inst.error ?? null;
		const memberCount = Object.keys(inst.members).length;
		// Everything the synth node RENDERS except position, which is applied in place below so a
		// per-frame drag keeps one identity and never churns the viewer.
		const labelSig = Object.entries(inst.interface)
			.map(([bid, p]) => `${bid}=${p.name ?? ''}`)
			.join(',');
		const sig = `${inst.name}|${error ?? ''}|${memberCount}|${JSON.stringify(inst.slots)}|${labelSig}`;

		const cached = this._synthCache.get(instId);
		if (cached && cached.sig === sig) {
			cached.node.pos = inst.pos; // keep position fresh without a new identity
			cached.node.viewers = inst.viewers ?? {}; // …and its view state, which the sig cannot carry
			return cached.node;
		}

		// External ports ARE the server-computed slots (a pure passthrough).
		const input_slots: Record<string, string> = { ...inst.slots.input };
		const output_slots: Record<string, string> = { ...inst.slots.output };
		// Keyed by the stable boundary id but labelled with the renameable portal NAME, so a rename
		// relabels the collapsed port without re-keying the wire.
		const slot_labels: Record<string, string> = {};
		for (const [bid, port] of Object.entries(inst.interface)) {
			if (port.name) slot_labels[bid] = port.name;
		}
		const node: NodeInstanceInfo = {
			uid: instId,
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

	/** Create a batch of nodes and wire the given links onto the new uids, returning the original→new
	 * map. A per-item failure is swallowed so one bad item does not abort the batch. */
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

	/** Duplicate nodes by uid, offset from the originals, carrying their params and inner links. */
	async cloneNodes(
		uids: Iterable<string>,
		offset: [number, number] = [40, 40],
		instId?: string
	): Promise<Record<string, string>> {
		const set = new Set(uids);
		// Link endpoints are uids, so the filter and the spec key must be uids too, or the rename
		// map will not line up with the link remap.
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

	/** Delete several nodes as ONE undoable step. */
	async removeNodes(uids: Iterable<string>): Promise<void> {
		const uidList = [...uids];
		if (uidList.length === 0) return;
		const label = `Delete ${uidList.length} node${uidList.length > 1 ? 's' : ''}`;
		// Each removeNode captures its OWN subtree, and a link rides with whichever endpoint owns
		// it, so delete order is immaterial.
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
