/** Control-plane WebSocket client: typed RPC and event subscription over `/control`. */
import type { ParamDescriptor } from '$lib/api/types';
import type { OpName } from '$lib/api/ops';

/** Control-plane protocol version. Bump it together with PROTOCOL_VERSION in
 * `backend/goofi-bridge/src/schemas.rs`. */
export const PROTOCOL_VERSION = 3;

/** Whether a backend-reported protocol version is compatible with this build. */
export function isProtocolCompatible(remote: unknown): boolean {
	return remote === PROTOCOL_VERSION;
}

/** A node's lifecycle stage. 'error' is terminal: the backend does not auto-restart it. */
export type NodeStage = 'creating' | 'setup' | 'ready' | 'error';

export interface NodeTypeInfo {
	type: string;
	category: string;
	/** Which tree the type came from; an `--extra-nodes` directory reads as `builtin`. */
	source: 'builtin' | 'patch';
	doc: string;
	/** Whether this machine resolves the type's unconditional top-level deps. */
	available: boolean;
	missing_deps: string[];
	input_slots: Record<string, string>;
	/** Names of the variadic (multi) input slots. */
	input_multi?: string[];
	output_slots: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
}

/** What one `rescan_nodes` changed, by type name. */
export interface ScanDiff {
	added: string[];
	changed: string[];
	removed: string[];
}

/** A node's self-reported execution telemetry. */
export interface NodeStats {
	updates_per_second: number;
}

export interface NodeInstanceInfo {
	/** The universal node identity, stable across rename/restart/reload. */
	uid: string;
	/** Mutable display name — the label only, never an identity key. */
	name: string;
	type: string;
	category: string;
	doc: string;
	input_slots: Record<string, string>;
	/** Names of the variadic (multi) input slots. */
	input_multi?: string[];
	output_slots: Record<string, string>;
	/** Optional per-slot display label, keyed by slot id; the slot id is shown when absent. */
	slot_labels?: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
	pos: [number, number];
	/** Per-output-slot view state restored from the .gfi patch. */
	viewers: Record<string, { collapsed?: boolean; kind?: string; settings?: Record<string, unknown> }>;
	/** Sub-patch membership marker; null for a top-level node. */
	membership?: { instance: string; local_name: string } | null;
	error: string | null;
	/** Lifecycle stage. Optional so a synthesized virtual node can omit it. */
	stage?: NodeStage;
	/** Rolling execution telemetry, absent until the node's first `node_stats` event. */
	stats?: NodeStats | null;
	/** Set only on a virtual node synthesized for a sub-patch scope. */
	subpatch?: SubpatchMeta | null;
}

/** Marks a virtual node as standing in for a sub-patch scope. */
export interface SubpatchMeta {
	instId: string;
	memberCount: number;
}

export interface LinkInfo {
	/** Source / target node UIDs (not display names). */
	node_out: string;
	node_in: string;
	slot_out: string;
	slot_in: string;
}

/** One boundary (In/Out node) of a sub-patch; `inner_node`/`inner_slot` are null when unwired. */
export interface SubPatchPort {
	dir: 'in' | 'out';
	dtype?: string;
	inner_node: string | null;
	inner_slot: string | null;
	pos: [number, number];
	/** The portal's renameable display label, decoupled from the stable routing key. */
	name?: string;
}

/** The category the six virtual In/Out types wear in the backend's palette. They are node types
 * like any other — `add_node` creates one, and `inst_id` says which sub-patch it is a port OF. */
export const BOUNDARY_CATEGORY = 'boundary';

/** The one slot a boundary port carries. MUST match the bridge's `vocab::BOUNDARY_SLOT`. */
export const BOUNDARY_SLOT = 'value';

/** A sub-patch instance the editor renders as a group node — mirrored from the bridge's
 * `describe_instance`, never re-derived here. */
export interface InstanceInfo {
	/** The scope's stable uid — also its facade node's uid. */
	uid: string;
	name: string;
	/** Parent scope uid; null at the top level. */
	parent: string | null;
	/** stub id -> port; `inner_node` is the direct inner member uid. */
	interface: Record<string, SubPatchPort>;
	pos: [number, number];
	/** member uid -> whether that member is itself a nested scope. */
	members: Record<string, { uid: string; is_instance: boolean }>;
	/** External ports computed from WIRED stubs: stub id -> dtype. */
	slots: { input: Record<string, string>; output: Record<string, string> };
	/** First errored descendant across the whole subtree, or null. */
	error: string | null;
	/** Per-output-boundary view state persisted in the .gfi patch, keyed by boundary id. */
	viewers: Record<string, { collapsed?: boolean; kind?: string; settings?: Record<string, unknown> }>;
}

/** Canonical string key for a link — its four slot endpoints. */
export function linkKey(l: LinkInfo): string {
	return `${l.node_out}.${l.slot_out}→${l.node_in}.${l.slot_in}`;
}


export interface FsEntry {
	name: string;
	path: string;
	kind: 'dir' | 'file';
	is_gfi: boolean;
	hidden: boolean;
}

export interface FsRoot {
	label: string;
	path: string;
}

export interface DirListing {
	path: string;
	parent: string | null;
	entries: FsEntry[];
	roots: FsRoot[];
}

/** Flatten a node's grouped param descriptors to a plain `{group: {name: value}}` bag. */
export function paramValues(node: NodeInstanceInfo): Record<string, Record<string, unknown>> {
	const out: Record<string, Record<string, unknown>> = {};
	for (const [group, params] of Object.entries(node.params)) {
		out[group] = {};
		for (const [name, p] of Object.entries(params)) out[group][name] = p.value;
	}
	return out;
}

export interface GraphSnapshot {
	/** Control-plane protocol version, present on the `hello` handshake. */
	protocol_version?: number;
	/** Identifies the manager process; it changes when the backend is restarted. */
	instance_id: string;
	/** Per-node runtime state, seeded here because its live stream pushes only transitions. */
	runtime: Record<string, { stage?: NodeStage; error?: string | null }>;
	/** The node palette, carried on `hello`/`graph_replaced`. Absent on an older backend. */
	node_types?: NodeTypeInfo[];
	save_path: string | null;
	unsaved_changes: boolean;
	/** Where THIS client was last looking — persisted with the patch, never converged to a peer. */
	viewpoint?: unknown;
	/** The spawned agent harnesses and the installed ones. Absent on an older backend. */
	harnesses?: HarnessRoster;
}

/** One spawned harness. `stopping` spans the grace period between the stop and the exit. */
export interface HarnessInstanceInfo {
	id: string;
	harness: string;
	state: 'running' | 'stopping' | 'exited';
	exit_code?: number | null;
}

/** One harness binary found on this machine. */
export interface DetectedHarness {
	harness: string;
	path: string;
	version: string | null;
}

/** The shape the snapshot seeds and `harness_changed` broadcasts. */
export interface HarnessRoster {
	instances: HarnessInstanceInfo[];
	detected: DetectedHarness[];
}

export type ControlEvent =
	| { event: 'hello'; payload: GraphSnapshot }
	// The node itself arrives via the doc; this carries no projection of it.
	| { event: 'node_added'; payload: { uid: string } }
	| {
			event: 'state_update';
			payload: {
				node: string;
				params: Record<string, Record<string, ParamDescriptor>>;
				stage?: NodeStage;
				// Re-pushed on the state plane, so a lost first error still surfaces and a clear lifts it.
				error?: string | null;
				// Params whose ⟳ refresh completed on this push, so the UI can clear the spinner.
				refreshed_params?: [string, string][];
			};
	  }
	| { event: 'error'; payload: { node: string; error: string | null } }
	| { event: 'node_stage'; payload: { node: string; stage: NodeStage; error?: string } }
	| { event: 'node_stats'; payload: { node: string; stats: NodeStats } }
	// Applied surgically (only these params' `value`), never a wholesale params replace.
	| {
			event: 'param_values';
			payload: { node: string; values: Record<string, Record<string, number | string | boolean>> };
	  }
	| { event: 'unsaved_changes'; payload: { unsaved_changes: boolean } }
	| { event: 'save_path_changed'; payload: { save_path: string | null } }
	// The palette changed under an already-connected client; `hello` carries it to an arriving one.
	| { event: 'node_types'; payload: { types: NodeTypeInfo[] } }
	// Carries the WHOLE roster, so a client never has to diff transitions.
	| { event: 'harness_changed'; payload: HarnessRoster }
	| { event: 'graph_replaced'; payload: GraphSnapshot }
	// The whole document — on connect, and again to recover a client that lagged past the ring.
	| { event: 'doc_state'; payload: { v: number; doc: Record<string, unknown> } }
	// `from` is the version the delta applies TO, `v` the version it produces.
	| { event: 'doc_patch'; payload: { from: number; v: number; patch: Record<string, unknown> } };

type EventHandler = (ev: ControlEvent) => void;

type Pending = {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
};

/** Minimal structural surface of the control client — the seam a test fake substitutes for. */
export interface Control {
	/** This client's stable session tag; it scopes the manager's per-session undo history. */
	readonly session: string;
	call<T = unknown>(op: OpName, payload?: Record<string, unknown>): Promise<T>;
	on(fn: (ev: ControlEvent) => void): () => void;
	onConnect(fn: (c: boolean) => void): () => void;
}

/** This tab's stable command-session id, minted once per tab in `sessionStorage`. */
function readOrMintSession(): string {
	try {
		const KEY = 'goofi:session';
		let s = sessionStorage.getItem(KEY);
		if (!s) {
			s = crypto?.randomUUID?.() ?? `s${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
			sessionStorage.setItem(KEY, s);
		}
		return s;
	} catch {
		return `s${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
	}
}

export class ControlClient implements Control {
	private ws: WebSocket | null = null;
	private url: string;
	readonly session = readOrMintSession();
	private nextId = 1;
	private pending = new Map<number, Pending>();
	private handlers = new Set<EventHandler>();
	private connectListeners = new Set<(connected: boolean) => void>();
	private protocolListeners = new Set<(mismatch: boolean) => void>();
	private _connected = false;
	private _protocolMismatch = false;
	private retryMs = 250;

	constructor(url?: string) {
		const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
		this.url = url ?? `${proto}//${location.host}/control`;
	}

	connect(): void {
		this._open();
	}

	private _open(): void {
		if (this.ws) return;
		const ws = new WebSocket(this.url);
		this.ws = ws;

		ws.addEventListener('open', () => {
			this.retryMs = 250;
			this._setConnected(true);
		});
		ws.addEventListener('message', (e) => this._onMessage(e));
		ws.addEventListener('close', () => this._onClose());
	}

	private _onMessage(e: MessageEvent): void {
		let msg: unknown;
		try {
			msg = JSON.parse(e.data);
		} catch {
			return;
		}
		if (typeof msg !== 'object' || msg === null) return;
		const obj = msg as Record<string, unknown>;
		if ('id' in obj && typeof obj.id === 'number') {
			const id = obj.id;
			const pending = this.pending.get(id);
			if (!pending) return;
			this.pending.delete(id);
			if ('error' in obj) pending.reject(new Error(String(obj.error)));
			else pending.resolve(obj.result);
			return;
		}
		if ('event' in obj && typeof obj.event === 'string') {
			if (obj.event === 'hello') this._checkProtocol(obj.payload);
			for (const h of this.handlers) {
				try {
					h(msg as ControlEvent);
				} catch (err) {
					console.error('control handler crashed', err);
				}
			}
		}
	}

	/** Latch a protocol mismatch from `hello`. One-way: only a reload clears it. */
	private _checkProtocol(payload: unknown): void {
		const remote =
			payload && typeof payload === 'object'
				? (payload as { protocol_version?: unknown }).protocol_version
				: undefined;
		if (this._protocolMismatch || isProtocolCompatible(remote)) return;
		this._protocolMismatch = true;
		for (const h of this.protocolListeners) h(true);
	}

	private _onClose(): void {
		this.ws = null;
		this._setConnected(false);
		for (const [, p] of this.pending) p.reject(new Error('control socket closed'));
		this.pending.clear();
		const delay = this.retryMs;
		this.retryMs = Math.min(this.retryMs * 2, 5000);
		setTimeout(() => this._open(), delay);
	}

	private _setConnected(v: boolean): void {
		if (this._connected === v) return;
		this._connected = v;
		for (const h of this.connectListeners) h(v);
	}

	/** Subscribe to all incoming events. Returns an unsubscribe fn. */
	on(handler: EventHandler): () => void {
		this.handlers.add(handler);
		return () => this.handlers.delete(handler);
	}

	onConnect(handler: (connected: boolean) => void): () => void {
		this.connectListeners.add(handler);
		handler(this._connected);
		return () => this.connectListeners.delete(handler);
	}

	/** Subscribe to a protocol-version mismatch (fires once, immediately if already mismatched). */
	onProtocolMismatch(handler: (mismatch: boolean) => void): () => void {
		this.protocolListeners.add(handler);
		if (this._protocolMismatch) handler(true);
		return () => this.protocolListeners.delete(handler);
	}

	/** Issue an RPC. Returns a promise resolving to the server's result. */
	call<T = unknown>(op: OpName, payload: Record<string, unknown> = {}): Promise<T> {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			return Promise.reject(new Error('control socket not connected'));
		}
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
			// `session` rides at the top level: the manager scopes its undo/redo history by it.
			this.ws!.send(JSON.stringify({ id, op, payload, session: this.session }));
		});
	}
}

/** Process-wide singleton (one bridge ↔ one tab). */
let _client: ControlClient | null = null;
export function getControl(): ControlClient {
	if (!_client) {
		_client = new ControlClient();
		_client.connect();
	}
	return _client;
}
