/** Central reactive graph state, backed by the control WS. The store owns the only writes, so a
 * component just reads its `$state` fields. */
import {
	getControl,
	type Control,
	type ControlEvent,
	type DirListing,
	type GraphSnapshot,
	type LinkInfo,
	type NodeInstanceInfo,
	type NodeTypeInfo,
	type ScanDiff
} from '$lib/api/control';
import { boundaryType } from '$lib/api/vocab';
import { bareName } from '$lib/editor/typeId';
import { consoleStore } from './console.svelte';
import { selection } from './selection.svelte';
import { workspace } from 'panelty';
import type { SlotView } from '$lib/viewers/inlineView';
import { history, type Action } from './history.svelte';
import { captureNavContext } from '$lib/stores/navContext';
import { SyncClient } from '$lib/crdt/syncClient';
import {
	linkViews,
	nodeViews,
	facadeFaces,
	docParams,
	viewersJson,
	globalViews,
	arrangementTabs,
	type Doc,
	type GlobalView,
	type GlobalType
} from '$lib/crdt/graphDoc';
import { assembleNode, type RuntimeOverlay } from '$lib/crdt/nodeAssembly';
import type { StringParam, SourcePatch } from '$lib/api/types';
import type { GraphFragment } from '$lib/editor/clipboard';

/** Safety net: lift a ⟳ spinner after this long when a node never reports the refresh done.
 * Generous — an LSL resolve blocks the node's ctrl thread ~4s. */
const REFRESH_SPINNER_TIMEOUT_MS = 15000;

/** Stable key for an in-flight param refresh; U+001F cannot occur in a uid/group/name. */
function refreshKey(node: string, group: string, name: string): string {
	return `${node}\u001f${group}\u001f${name}`;
}

/** A doc link as the wire ops spell it: two `uid/slot` endpoints. */
function linkEndpoints(link: LinkInfo): { from: string; to: string } {
	return {
		from: `${link.node_out}/${link.slot_out}`,
		to: `${link.node_in}/${link.slot_in}`
	};
}

export class GraphStore {
	nodes = $state<NodeInstanceInfo[]>([]);
	links = $state<LinkInfo[]>([]);
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

	/** Per-node runtime for nodes the runtime planes named but the doc has not yet materialized.
	 * One-shot: `_seedRuntime` takes its entry, so a later node at the same uid seeds fresh. */
	private _snapshotRuntime: GraphSnapshot['runtime'] = {};

	/** Hold a runtime plane's report for a node still to materialize — the stage and error planes
	 * ride their own channel in no defined order against the doc, so a report outrunning the doc
	 * is routine, and dropping it would wedge the seed's `creating` guess for good. */
	private _stashRuntime(uid: string, rt: GraphSnapshot['runtime'][string]): void {
		this._snapshotRuntime[uid] = { ...this._snapshotRuntime[uid], ...rt };
	}

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
		// No-ops until the catalog lands, then rebuilds from the doc.
		this._reconcileNodesFromDoc();
	}

	/** Apply a wholesale snapshot, returning whether it came from a NEW backend session — which is
	 * what a same-session reconnect must not look like. */
	private _replaceSnapshot(snap: GraphSnapshot): boolean {
		// A `hello` always carries the palette; `graph_replaced` never does — the `node_types` event
		// is what re-announces it there.
		if (snap.node_types?.length) this.nodeTypes = snap.node_types;
		// The snapshot and the doc delta ride separate channels in no defined order, so the runtime
		// overlay is both stashed for nodes still to materialize and applied to those already here.
		this._snapshotRuntime = snap.runtime ?? {};
		for (const [uid, rt] of Object.entries(this._snapshotRuntime)) {
			const node = this.nodeById(uid);
			if (!node) continue;
			node.stage = rt.stage;
			node.error = rt.error ?? null;
			node.runtime = rt.runtime;
		}
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
	 * pick. A facade and a boundary port take the same op a leaf does — each is a node here. */
	setSlotView(uid: string, slot: string, view: SlotView): void {
		const node = this.nodeById(uid);
		if (!node?.output_slots[slot]) return;
		void this.ctl
			.call('node edit', { node: uid, viewer: [{ slot, ...view }] })
			.then(() => this._recordGraphCmd(`Set ${slot} view`))
			.catch(() => {
				/* soft view state — the next edit re-sends it */
			});
	}

	private _handle(ev: ControlEvent): void {
		switch (ev.event) {
			case 'hello': {
				// Not wholesale: a `hello` is also what a transient reconnect delivers.
				const fresh = this._replaceSnapshot(ev.payload);
				this.hadHello = true;
				if (fresh) {
					// A NEW session mints uids from 1 again, so the stale replica must fall NOW,
					// synchronously, before this connection answers the server's binary hello SV.
					// Projections first: `_resetProjection` reads `this.nodes`.
					this._resetProjection();
					this._sync.reset();
					this._onWholesaleLoad();
				}
				break;
			}
			case 'graph_replaced':
				this._replaceSnapshot(ev.payload);
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
					if ('error' in ev.payload) t.error = ev.payload.error ?? null;
					if (ev.payload.runtime !== undefined) t.runtime = ev.payload.runtime ?? undefined;
				}
				// Lift each spinner exactly when the fresh options land. Keyed by node, not by `t`.
				for (const [group, name] of ev.payload.refreshed_params ?? []) {
					this._endRefresh(refreshKey(ev.payload.node, group, name));
				}
				break;
			}
			case 'node_stage': {
				const t = this.nodeById(ev.payload.node);
				if (t) {
					t.stage = ev.payload.stage;
					if (ev.payload.error !== undefined) t.error = ev.payload.error ?? null;
					// The tier arrives here as well as on the snapshot: a node added after connecting
					// is in no snapshot, and a GIL demotion moves it while the session is live.
					if (ev.payload.runtime !== undefined) t.runtime = ev.payload.runtime ?? undefined;
				} else {
					// The stage plane pushes each transition ONCE, so one outrun by its node's own
					// doc delta is stashed for the seed, never dropped.
					this._stashRuntime(ev.payload.node, {
						stage: ev.payload.stage,
						error: ev.payload.error,
						runtime: ev.payload.runtime ?? undefined
					});
				}
				break;
			}
			case 'node_stats': {
				const t = this.nodeById(ev.payload.node);
				if (t) t.stats = ev.payload.stats;
				break;
			}
			case 'param_values': {
				// The node's WHOLE live-value map, never a delta: a driven param it no longer names
				// has no live value and shows its literal again.
				const t = this.nodeById(ev.payload.node);
				if (t) {
					const runtime = this._extractRuntime(t);
					for (const [group, names] of Object.entries(runtime.params ?? {})) {
						for (const [name, pr] of Object.entries(names)) {
							const live = ev.payload.values[group]?.[name];
							if (live === undefined) delete pr.liveValue;
							else pr.liveValue = live;
						}
					}
					this._reassembleNode(t, runtime);
				}
				break;
			}
			case 'error': {
				// A REPORT, so only a node that RUNS raises one — a facade's health rides `node_stage`.
				const t = this.nodeById(ev.payload.node);
				if (t) t.error = ev.payload.error;
				else this._stashRuntime(ev.payload.node, { error: ev.payload.error });
				if (ev.payload.error)
					consoleStore().ingestError(ev.payload.node, ev.payload.error, Date.now());
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

	/** Adopt a palette catalog. It supplies the descriptors, so the nodes rebuild here. */
	private _applyNodeTypes(types: NodeTypeInfo[]): void {
		this.nodeTypes = types;
		this._reconcileNodesFromDoc();
	}

	/** Re-derive the node registry from disk and report what changed; explicit, since there is no
	 * watcher. The fresh catalog arrives as a `node_types` event in every open tab. */
	async rescanNodes(): Promise<ScanDiff> {
		return this.ctl.call<ScanDiff>('library refresh', {});
	}

	/** Where this patch's workspace files live — a per-run temp directory under a random name. It
	 * rides `session status` beside the save path, because both answer "where does this patch live". */
	async openWorkspace(): Promise<string> {
		const r = await this.ctl.call<{ workspace: string }>('session status', {});
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

	async addNode(type: string, pos: [number, number], instId?: string): Promise<string> {
		const born = await this.ctl.call<{ uid: string }>('node add', {
			type,
			pos,
			inst_id: instId
		});
		const uid = born?.uid ?? '';
		if (uid) this._recordGraphCmd(`Add ${bareName(type)}`);
		return uid;
	}

	async removeNode(uid: string): Promise<void> {
		// The label reads the node's name before it vanishes.
		const label = `Delete ${this.nodeById(uid)?.name ?? uid}`;
		await this.ctl.call('node remove', { node: uid });
		this._recordGraphCmd(label);
	}

	/** Respawn a node in place, keeping its uid, name, params, position, scope and links. A
	 * recovery action rather than an edit, so it records no history. */
	async restartNode(uid: string): Promise<void> {
		await this.ctl.call('node restart', { node: uid });
	}

	/** Show a node's own editor window, or hide it. It opens on the machine the SERVER runs on —
	 * see the op's own words — so this is a request, not something the page can draw. */
	async showNodeEditor(uid: string, show: boolean): Promise<void> {
		await this.ctl.call('node editor', { node: uid, show });
	}

	async addLink(link: LinkInfo): Promise<void> {
		await this.ctl.call('link add', linkEndpoints(link));
		this._recordGraphCmd('Connect');
	}

	async removeLink(link: LinkInfo): Promise<void> {
		await this.ctl.call('link remove', linkEndpoints(link));
		this._recordGraphCmd('Disconnect');
	}

	async updateParam(node: string, group: string, name: string, value: unknown): Promise<void> {
		// Guard on EXISTENCE, not truthiness — a real param may hold 0, false or ''.
		const param = this.nodeById(node)?.params?.[group]?.[name];
		if (!param) throw new Error(`node param edit: no param ${group}.${name} on node ${node}`);
		await this.ctl.call('node param edit', { node, param: `${group}/${name}`, value });
		this._recordGraphCmd(`Set ${name}`);
	}

	/** Add a NEW user global; the server refuses a name the patch already holds. */
	async addGlobal(name: string, value: number | string | boolean, type: GlobalType): Promise<void> {
		if (this.globals.some((g) => g.name === name)) throw new Error(`global ${name} already exists`);
		await this.ctl.call('global add', { name, value, type });
		this._recordGraphCmd(`Add global ${name}`);
	}

	/** Edit an existing global's value (system or user); the type is immutable and stays. */
	async setGlobalValue(name: string, value: number | string | boolean): Promise<void> {
		if (!this.globals.some((g) => g.name === name)) throw new Error(`no global ${name}`);
		await this.ctl.call('global edit', { name, value });
		this._recordGraphCmd(`Set global ${name}`);
	}

	/** Remove a user global (a system global is refused by the server). */
	async removeGlobal(name: string): Promise<void> {
		await this.ctl.call('global remove', { name });
		this._recordGraphCmd(`Remove global ${name}`);
	}

	/** Rename a user global; refs are NOT rewritten, so a stale `globals.<old>` throws at eval time.
	 * A set of the new name compounded with a delete of the old, so it is one undo step. */
	async renameGlobal(oldName: string, newName: string): Promise<void> {
		const held = this.globals.find((g) => g.name === oldName);
		if (!held) throw new Error(`no global ${oldName}`);
		await this.ctl.call('compound', {
			ops: [
				{ op: 'global add', payload: { name: newName, value: held.value, type: held.type } },
				{ op: 'global remove', payload: { name: oldName } }
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
			await this.ctl.call('node param refresh', { node, param: `${group}/${name}` });
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

	/** Edit a param's source record: any subset of mode, expression, reference and triggers. A text
	 * given implies its mode; an empty text clears it. The manager's rules are the op's. */
	async setSource(node: string, group: string, name: string, source: SourcePatch): Promise<void> {
		const d = this.nodeById(node)?.params?.[group]?.[name];
		if (!d) throw new Error(`node param edit: no param ${group}.${name} on node ${node}`);
		await this.ctl.call('node param edit', { node, param: `${group}/${name}`, ...source });
		this._recordGraphCmd(`Set ${name} source`);
	}

	async setNodePos(uid: string, pos: [number, number]): Promise<void> {
		// Committed on drag-stop only; a live drag stays local to Svelte Flow.
		await this.ctl.call('node edit', { node: uid, pos });
		this._recordGraphCmd(`Move ${this.nodeById(uid)?.name ?? uid}`);
	}

	/** Set a node's mutable display name (uid identity is unchanged). */
	async renameNode(uid: string, name: string): Promise<void> {
		const oldName = this.nodeById(uid)?.name ?? '';
		if (oldName === name) return;
		await this.ctl.call('node edit', { node: uid, name });
		this._recordGraphCmd(`Rename ${oldName} → ${name}`);
	}

	/** Store where THIS client is looking. Persisted in the `.gfi`, but never converged and never
	 * dirtying: persistence and dirtiness are separate axes. */
	async setViewpoint(viewpoint: unknown): Promise<void> {
		try {
			await this.ctl.call('layout viewpoint edit', { value: viewpoint });
		} catch {
			/* not connected / in flight — ignore */
		}
	}

	/** Write the patch. Where it landed comes back from the MANAGER (`save_path_changed`), never
	 * latched from this reply — a latch names the patch only in the tab that saved it. */
	async save(path: string): Promise<{ path: string }> {
		// A given `path` becomes the patch's home; the arrangement is the manager's already.
		return this.ctl.call<{ path: string }>('session save', { path });
	}

	/** Reset to an empty, unnamed patch. Nothing is written here: a New emits no
	 * `save_path_changed`, so the `graph_replaced` snapshot is the sole carrier of the null path. */
	async newPatch(): Promise<void> {
		await this.ctl.call('session new', {});
	}

	/** Group the named nodes into a sub-patch. Returns its instance id. */
	async groupNodes(members: string[], pos?: [number, number]): Promise<string> {
		const r = await this.ctl.call<{ inst_id: string }>('nodes group', { nodes: members, pos });
		if (r?.inst_id) this._recordGraphCmd('Group nodes');
		return r.inst_id;
	}

	/** Dissolve a sub-patch instance back into its member nodes. */
	async expandInstance(instId: string): Promise<void> {
		await this.ctl.call('nodes ungroup', { subpatch: instId });
		this._recordGraphCmd('Ungroup');
	}

	/** List one directory level on the BACKEND filesystem (full FS, no jail). */
	async listDir(path?: string): Promise<DirListing> {
		return this.ctl.call<DirListing>('dir list', { path });
	}

	/** Load a patch from a BACKEND filesystem path; destructive, and it resets the session, so
	 * there is no history entry. A `.gfi` is a zip, so a path is the only door the client has. */
	async load(path: string): Promise<void> {
		await this.ctl.call('session load', { path });
	}

	/** Resolve a node by uid — the ONE accessor, and every kind of node record answers it. */
	nodeById(id: string): NodeInstanceInfo | null {
		return this.nodes.find((n) => n.uid === id) ?? null;
	}

	/** Every node a panel can bind or a picker can list. ROOT is the canvas, and it is not one. */
	get bindable(): { uid: string; name: string }[] {
		return this.nodes.map((n) => ({ uid: n.uid, name: n.name }));
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

	/** The runtime overlay for a node materializing from the doc for the FIRST time. A leaf is
	 * `creating` until its own thread says otherwise; a virtual node runs nothing, so it is born at
	 * the stage the backend already answers for it rather than waiting a stats period to hear so. */
	private _seedRuntime(uid: string, virtual: boolean): RuntimeOverlay {
		const seed = this._snapshotRuntime[uid];
		delete this._snapshotRuntime[uid];
		return {
			stage: seed?.stage ?? (virtual ? 'ready' : 'creating'),
			error: seed?.error ?? null,
			runtime: seed?.runtime
		};
	}

	/** Pull the RUNTIME (event-sourced, never-in-the-doc) fields off a node so a re-assemble keeps them. */
	private _extractRuntime(node: NodeInstanceInfo): RuntimeOverlay {
		const params: NonNullable<RuntimeOverlay['params']> = {};
		for (const group of Object.keys(node.params)) {
			params[group] = {};
			for (const name of Object.keys(node.params[group])) {
				const p = node.params[group][name];
				const pr: NonNullable<RuntimeOverlay['params']>[string][string] = { error: p.error };
				if (p.type === 'string') pr.options = (p as StringParam).options;
				// A driven param DISPLAYS its live evaluated value, which never reaches the doc.
				if (p.mode !== 'constant') pr.liveValue = p.value;
				params[group][name] = pr;
			}
		}
		return {
			error: node.error,
			stage: node.stage,
			runtime: node.runtime,
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
				const d = desc as { error?: string | null; options?: string[] | null };
				(p as { error: string | null }).error = d.error ?? null;
				if (p.type === 'string') (p as StringParam).options = d.options ?? null;
			}
		}
	}

	/** Re-assemble ONE node from the doc under `runtime`, in place, so a runtime-only change reads
	 * the doc's literals through the same merge a doc change does. */
	private _reassembleNode(t: NodeInstanceInfo, runtime: RuntimeOverlay): void {
		const doc = this._sync.doc;
		const nv = nodeViews(doc).find((v) => v.uid === t.uid);
		if (!nv) return;
		const catalog = this.nodeTypes?.find((c) => c.type === nv.type);
		const faces = facadeFaces(doc);
		const viewers = (viewersJson(doc, t.uid) ?? {}) as NodeInstanceInfo['viewers'];
		Object.assign(t, assembleNode(nv, docParams(doc, t.uid), viewers, catalog, runtime, faces.get(t.uid)));
	}

	/** Build `this.nodes` from the doc: each record is the doc's own fields, plus the catalog
	 * descriptor for its type, plus the runtime overlay the doc never holds. A facade has no
	 * catalog entry — it runs nothing — so its ports supply its slots instead. */
	private _reconcileNodesFromDoc(): void {
		if (!this.nodeTypes?.length) return; // no catalog yet → keep the current nodes; rebuild when it lands
		const doc = this._sync.doc;
		// Both indexes are built ONCE per reconcile rather than per node.
		const byType = new Map(this.nodeTypes.map((t) => [t.type, t]));
		const faces = facadeFaces(doc);
		const next: NodeInstanceInfo[] = nodeViews(doc).map((nv) => {
			const existing = this.nodeById(nv.uid);
			const catalog = byType.get(nv.type);
			const runtime: RuntimeOverlay = existing
				? this._extractRuntime(existing)
				: this._seedRuntime(nv.uid, !!boundaryType(nv.type) || faces.has(nv.uid));
			const viewers = (viewersJson(doc, nv.uid) ?? {}) as NodeInstanceInfo['viewers'];
			return assembleNode(nv, docParams(doc, nv.uid), viewers, catalog, runtime, faces.get(nv.uid));
		});
		this._reconcileNodes(next);
	}

	/** Read `uids` and everything they hold — members, ports and nested sub-patches, to any depth —
	 * as a self-contained fragment in the shape a `.gfi` carries. */
	async copyNodes(uids: string[]): Promise<GraphFragment> {
		const r = await this.ctl.call<{ doc: GraphFragment }>('nodes copy', { nodes: uids });
		return r.doc;
	}

	/** Add a fragment on fresh uids, shifted by `offset` and rooted in `instId`. ONE command, so a
	 * paste of any depth is one undo step; answers each record's uid mapped to what it became. */
	async pasteNodes(
		doc: GraphFragment,
		offset: [number, number] = [0, 0],
		instId?: string
	): Promise<Record<string, string>> {
		const r = await this.ctl.call<{ rename: Record<string, string> }>('nodes paste', {
			doc,
			pos: offset,
			inst_id: instId ?? null
		});
		this._recordGraphCmd('Paste nodes');
		return r.rename ?? {};
	}

	/** Duplicate `uids` in place — a copy and a paste, which is what a duplicate IS, so a sub-patch
	 * and a leaf go through the one door. `instId` is where the selection came FROM: a fragment
	 * names a scope only when that scope is in it, so a member copied alone names none. */
	async cloneNodes(
		uids: string[],
		offset: [number, number] = [40, 40],
		instId?: string
	): Promise<Record<string, string>> {
		if (uids.length === 0) return {};
		return this.pasteNodes(await this.copyNodes(uids), offset, instId);
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
