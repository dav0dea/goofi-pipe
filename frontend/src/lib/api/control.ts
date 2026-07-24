/**
 * Control-plane WebSocket client.
 *
 * Connects to `/control`, exposes a typed RPC method (`call`), and an
 * event subscription mechanism (`on`). Auto-reconnects with exponential
 * backoff; pending RPCs that were in-flight when the socket dropped are
 * rejected so callers can re-issue if appropriate.
 */
import type { ParamDescriptor } from '$lib/api/types';

/** Control-plane protocol version. The browser reconciles purely from echoed
 * events, so a stale `frontend/build/` against a newer backend would diverge
 * SILENTLY rather than erroring. The backend stamps its version into `hello`
 * (`crates/goofi-bridge/src/schemas.rs` PROTOCOL_VERSION) and the client asserts a match —
 * turning silent skew into an explicit "reload required". Bump BOTH sides
 * together whenever the wire shape or reconciliation rules change. */
export const PROTOCOL_VERSION = 1;

/** Whether a backend-reported protocol version is compatible with this build.
 * Strict equality: an absent/non-numeric version means a backend older than this
 * field, which is itself a skew worth flagging. */
export function isProtocolCompatible(remote: unknown): boolean {
	return remote === PROTOCOL_VERSION;
}

/** A node's lifecycle stage: 'creating' (process spawned; importing its
 * implementation + opening endpoints) → 'setup' (first state arrived, setup()
 * still running) → 'ready'. 'error' is terminal (bootstrap failure — the
 * backend does not auto-restart it). */
export type NodeStage = 'creating' | 'setup' | 'ready' | 'error';

export interface NodeTypeInfo {
	type: string;
	category: string;
	doc: string;
	/** Unconditional top-level deps resolvable on this machine (registry probe).
	 * Unavailable types render greyed/disabled in the add menu. */
	available: boolean;
	/** Config hooks reference runtime state (device lists) — the palette shows
	 * the static declaration; options refresh from the live node. Informational. */
	dynamic: boolean;
	missing_deps: string[];
	input_slots: Record<string, string>;
	/** Names of the variadic (multi) input slots — static type shape the UI reads to
	 * render those slots tall and accept many cables. Read-only; never toggled here. */
	input_multi?: string[];
	output_slots: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
}

/** A node's self-reported execution telemetry (mirrors the backend NODE_STATS
 * payload). `updates_per_second` is the tick cadence; `mean_process_ms` the mean
 * `process()` wall-time over the window; `total_ticks` the lifetime count. */
export interface NodeStats {
	updates_per_second: number;
	/** Optional — not measured by the Rust engine yet, so omitted rather than faked. */
	mean_process_ms?: number;
	total_ticks?: number;
}

export interface NodeInstanceInfo {
	/** The UNIVERSAL node identity — the key everything references the node by
	 * (flow id, selection, links, data subscriptions, panel bindings). Stable
	 * across rename/restart/reload. For a sub-patch group node (a synthesized
	 * virtual node) this is the instance id; for a real node it's the backend uid. */
	uid: string;
	/** Mutable DISPLAY name (e.g. "oscillator0") — flat and globally unique at every
	 * nesting depth (no `inst::local` qualification). For the label only — never an
	 * identity key, so it can be renamed freely. */
	name: string;
	type: string;
	category: string;
	doc: string;
	input_slots: Record<string, string>;
	/** Names of the variadic (multi) input slots — static type shape the UI reads to
	 * render those slots tall and accept many cables. Read-only; never toggled here. */
	input_multi?: string[];
	output_slots: Record<string, string>;
	/** Optional per-slot DISPLAY label, keyed by slot id (the handle/routing key). When
	 * absent for a slot, the slot id is shown. Used by a sub-patch synth node to show a
	 * portal's renameable name while keeping the stable boundary id as the handle. */
	slot_labels?: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
	pos: [number, number];
	/** Per-output-slot view state restored from the .gfi patch — empty when
	 * the node was just spawned, populated when a save was previously made.
	 * `kind` is the chosen viewer type; `settings` are that viewer's cog-menu
	 * overrides. */
	viewers: Record<string, { collapsed?: boolean; kind?: string; settings?: Record<string, unknown> }>;
	/** Sub-patch membership marker; null for a top-level node. Members of a
	 * collapsed instance are hidden from the canvas (the group node stands in). */
	membership?: { instance: string; local_name: string } | null;
	error: string | null;
	/** Lifecycle stage — seeded by node_added/snapshot, updated by state_update
	 * and node_stage events. Drives the boot spinner ('creating'/'setup') and
	 * the terminal boot-error badge. Backend always sends it for real nodes;
	 * optional so synthesized virtual nodes (sub-patch instances) can omit it. */
	stage?: NodeStage;
	/** Rolling execution telemetry the node pushes on the status plane (~1 Hz):
	 * update rate + mean `process()` duration over the last ~10 ticks. Absent until
	 * the node's first NODE_STATS push; populated by the `node_stats` event and
	 * present in the snapshot for a freshly-connected client. Runtime UI state. */
	stats?: NodeStats | null;
	/** Set only on a *virtual* node synthesized for a sub-patch scope (see
	 * `graph.nodeById`). Lets the node layers — selection, inspector, drag —
	 * treat a sub-patch like a node while the inspector renders its controls
	 * instead of param groups. Absent/null for real nodes. */
	subpatch?: SubpatchMeta | null;
}

/** Marks a virtual node as standing in for a sub-patch scope. Only the id rides
 * along — member count is recomputed from the `instances` map so it never goes stale. */
export interface SubpatchMeta {
	instId: string;
	/** Number of member nodes (shown as a badge on the collapsed group node). */
	memberCount: number;
}

export interface LinkInfo {
	/** Source / target node UIDs (not display names). */
	node_out: string;
	node_in: string;
	slot_out: string;
	slot_in: string;
}

/** One boundary (In/Out node) of a sub-patch. `inner_node`/`inner_slot` are null
 * when the boundary is UNWIRED (added but not yet connected to a member); a wired
 * boundary becomes a port on the collapsed group node. `pos` is the In/Out pill's
 * position inside the entered view. */
export interface SubPatchPort {
	dir: 'in' | 'out';
	dtype?: string;
	inner_node: string | null;
	inner_slot: string | null;
	pos: [number, number];
	/** The portal's renameable DISPLAY/slot label (defaults to in0/out0). Decoupled from
	 * the interface KEY, which stays the stable routing id — so a rename never re-keys an
	 * external wire. */
	name?: string;
}

/** The six virtual In/Out node types — one per data type per direction. They are
 * placeable ONLY inside a sub-patch (the add-menu surfaces them when entered) and
 * carry no params; the type name encodes the boundary's direction + data type. */
export interface BoundarySpec {
	dir: 'in' | 'out';
	dtype: string;
}
const BOUNDARY_DTYPES = ['Array', 'String', 'Table'] as const;
export const BOUNDARY_TYPES: NodeTypeInfo[] = (['In', 'Out'] as const).flatMap((side) =>
	BOUNDARY_DTYPES.map((dt): NodeTypeInfo => {
		const slot: Record<string, string> = { value: dt.toUpperCase() };
		return {
			type: `${side}${dt}`,
			category: 'boundary',
			doc: `Sub-patch ${side === 'In' ? 'input' : 'output'} (${dt.toLowerCase()})`,
			// An In node feeds a member input (it has an OUTPUT of the dtype); an Out
			// node drains a member output (it has an INPUT). Declaring these lets the
			// seeded add-menu's dtype filter keep In/Out as valid auto-connect targets
			// when you click a member slot inside a sub-patch.
			input_slots: side === 'Out' ? slot : {},
			output_slots: side === 'In' ? slot : {},
			params: {},
			// Virtual types have no implementation module — always addable.
			available: true,
			dynamic: false,
			missing_deps: []
		};
	})
);

/** Parse a boundary pseudo-type name (e.g. "InArray") to its dir + dtype, or null. */
export function boundarySpec(type: string): BoundarySpec | null {
	const m = /^(In|Out)(Array|String|Table)$/.exec(type);
	if (!m) return null;
	return { dir: m[1] === 'In' ? 'in' : 'out', dtype: m[2].toUpperCase() };
}

/** A flatten-at-runtime sub-patch instance the editor renders as a group node.
 * This is a server-COMPUTED record (see bridge `describe_instance`) that the frontend
 * MIRRORS — every field below is computed once on the backend, never re-derived here. */
export interface InstanceInfo {
	/** The scope's stable uid (also its key in the instances map, and its facade node's uid). */
	uid: string;
	/** Display label, e.g. "subpatch0" — separate from the uid key. */
	name: string;
	/** Parent scope uid (the nesting tree edge); null at the top level. */
	parent: string | null;
	/** stub id -> port (the scope's boundary stubs; `inner_node` is the DIRECT inner member uid). */
	interface: Record<string, SubPatchPort>;
	pos: [number, number];
	/** member uid -> { uid, whether the member is itself a nested scope } (flat model: keyed by
	 * uid, no template-local names) — lets the editor split direct children into plain nodes vs
	 * nested collapsed sub-patches. */
	members: Record<string, { uid: string; is_instance: boolean }>;
	/** External ports computed from WIRED stubs: stub id -> dtype. Mirror the collapsed facade
	 * node's input/output slots (a pure passthrough). */
	slots: { input: Record<string, string>; output: Record<string, string> };
	/** First errored DESCENDANT across the whole subtree (recursion-correct), or null. */
	error: string | null;
	/** Per-output-boundary view state persisted in the .gfi patch (round-trips), keyed
	 * by boundary id — same shape as a node's `viewers`. */
	viewers: Record<string, { collapsed?: boolean; kind?: string; settings?: Record<string, unknown> }>;
}

/** Canonical string key for a link — its four slot endpoints. Stable and
 * unique per link, usable as a map key or the SvelteFlow edge id. */
export function linkKey(l: LinkInfo): string {
	return `${l.node_out}.${l.slot_out}→${l.node_in}.${l.slot_in}`;
}


export interface FsEntry {
	name: string;
	path: string;
	kind: 'dir' | 'file';
	is_gfi: boolean;
	hidden: boolean;
	size: number;
	mtime: number;
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

/** Flatten a node's grouped param descriptors to a plain `{group: {name: value}}`
 * bag — the shape clone/clipboard/replay all need. */
export function paramValues(node: NodeInstanceInfo): Record<string, Record<string, unknown>> {
	const out: Record<string, Record<string, unknown>> = {};
	for (const [group, params] of Object.entries(node.params)) {
		out[group] = {};
		for (const [name, p] of Object.entries(params)) out[group][name] = p.value;
	}
	return out;
}

export interface GraphSnapshot {
	/** Control-plane protocol version, present on the `hello` handshake. The
	 * client asserts it against {@link PROTOCOL_VERSION}; a mismatch surfaces a
	 * "reload required" state instead of silently diverging. */
	protocol_version?: number;
	/** Identifies the manager *process*. Changes when the backend is restarted,
	 * letting the graph store distinguish a fresh session (reset the layout)
	 * from a transient reconnect to the same one (keep the current layout). */
	instance_id: string;
	/** Per-node RUNTIME state (`{uid: {stage, error}}`) — the event-sourced overlay the CRDT doc
	 * never holds. Seeded here because its live stream pushes only transitions, so a client that
	 * connects to a *running* graph would otherwise render an errored node as healthy. The graph
	 * STRUCTURE is deliberately absent: nodes, links and the sub-patch forest live in the doc. */
	runtime: Record<string, { stage?: NodeStage; error?: string | null }>;
	/** The node palette (same list `list_nodes` returns), carried on `hello` /
	 * `graph_replaced` so the client has descriptors in hand immediately — no async
	 * round-trip before it can build nodes from the CRDT doc. Absent on an older
	 * backend (the client then falls back to fetching `list_nodes`). */
	node_types?: NodeTypeInfo[];
	save_path: string | null;
	unsaved_changes: boolean;
	/** Opaque frontend workspace-layout blob restored from the .gfi patch.
	 * Null/absent when the patch carries no layout (older patch, or blank
	 * start) — a fresh session then falls back to the default layout. */
	layout?: unknown;
}

export type ControlEvent =
	| { event: 'hello'; payload: GraphSnapshot }
	// A bare "a node was created" announcement (the node itself arrives via the doc). Kept as the
	// one hook a client can act on at creation time; carries no projection of the node.
	| { event: 'node_added'; payload: { uid: string } }
	| {
			event: 'state_update';
			payload: {
				node: string;
				params: Record<string, Record<string, ParamDescriptor>>;
				stage?: NodeStage;
				// The node's current error, carried on the (always-on, re-pushed) state
				// plane so a lost first PROCESSING_ERROR still surfaces and a healthy
				// respawn's clear (null) lifts a stale chip.
				error?: string | null;
				// Params (`[group, name]`) whose ⟳ refresh just completed on this push —
				// the node re-scanned its options and reports done (success or failure),
				// so the UI can clear the per-param spinner exactly when fresh options land.
				refreshed_params?: [string, string][];
			};
	  }
	| { event: 'error'; payload: { node: string; error: string | null } }
	| { event: 'node_stage'; payload: { node: string; stage: NodeStage; error?: string } }
	| { event: 'node_stats'; payload: { node: string; stats: NodeStats } }
	// Live evaluated values of a node's expression-driven params (`{group: {name: value}}`),
	// pushed at the stats cadence so the inspector preview tracks each re-evaluation. Applied
	// surgically (only these params' `value`), never a wholesale params replace.
	| {
			event: 'param_values';
			payload: { node: string; values: Record<string, Record<string, number | string | boolean>> };
	  }
	| { event: 'unsaved_changes'; payload: { unsaved_changes: boolean } }
	| { event: 'save_path_changed'; payload: { save_path: string | null } }
	| { event: 'graph_replaced'; payload: GraphSnapshot }
	| { event: 'layout'; payload: { layout: unknown } };

type EventHandler = (ev: ControlEvent) => void;

type Pending = {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
};

/** Minimal structural surface of the control client — the dependency-injection
 * seam that executors and the history store depend on, so unit tests can
 * substitute a fake (see `$lib/test/fakeControl`). `ControlClient` implements
 * it; nothing else needs to. */
export interface Control {
	call<T = unknown>(op: string, payload?: Record<string, unknown>): Promise<T>;
	on(fn: (ev: ControlEvent) => void): () => void;
	onConnect(fn: (c: boolean) => void): () => void;
	/** Subscribe to inbound binary CRDT sync frames. Returns an unsubscribe fn. */
	onSyncFrame(fn: (bytes: Uint8Array) => void): () => void;
	/** Send an outbound binary CRDT sync frame (no-op if the socket is not open). */
	sendSync(bytes: Uint8Array): void;
}

/** This tab's stable command-session id (scopes the manager's per-client undo/redo). Minted once
 * per tab in `sessionStorage` so it survives WS reconnects but is unique across tabs; a fallback
 * id covers a non-browser context (SSR/tests) where `sessionStorage` is absent. */
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
	/** This tab's undo/redo session tag, sent on every request (see {@link readOrMintSession}). */
	readonly session = readOrMintSession();
	private nextId = 1;
	private pending = new Map<number, Pending>();
	private handlers = new Set<EventHandler>();
	private connectListeners = new Set<(connected: boolean) => void>();
	private protocolListeners = new Set<(mismatch: boolean) => void>();
	private syncListeners = new Set<(bytes: Uint8Array) => void>();
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
		// Binary frames carry CRDT sync updates; receive them as ArrayBuffer, not Blob.
		ws.binaryType = 'arraybuffer';
		this.ws = ws;

		ws.addEventListener('open', () => {
			this.retryMs = 250;
			this._setConnected(true);
		});
		ws.addEventListener('message', (e) => this._onMessage(e));
		ws.addEventListener('close', () => this._onClose());
	}

	private _onMessage(e: MessageEvent): void {
		// Binary frames are CRDT sync updates (ArrayBuffer, since binaryType is set) — route
		// them to the sync layer, which drives the Yjs replica. Text frames are JSON RPC.
		if (e.data instanceof ArrayBuffer) {
			const bytes = new Uint8Array(e.data);
			for (const h of this.syncListeners) {
				try {
					h(bytes);
				} catch (err) {
					console.error('sync frame handler crashed', err);
				}
			}
			return;
		}
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

	/** Latch a protocol mismatch from the `hello` handshake (one-way: a reload is
	 * the only fix, so we never clear it). Notifies subscribers once. */
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

	/** Subscribe to a protocol-version mismatch (fires once, immediately if already
	 * mismatched). A mismatch means this build can't safely talk to the running
	 * backend — the UI should prompt a reload. */
	onProtocolMismatch(handler: (mismatch: boolean) => void): () => void {
		this.protocolListeners.add(handler);
		if (this._protocolMismatch) handler(true);
		return () => this.protocolListeners.delete(handler);
	}

	/** Subscribe to inbound binary CRDT sync frames. Returns an unsubscribe fn. */
	onSyncFrame(fn: (bytes: Uint8Array) => void): () => void {
		this.syncListeners.add(fn);
		return () => this.syncListeners.delete(fn);
	}

	/** Send an outbound binary CRDT sync frame (dropped if the socket is not open). */
	sendSync(bytes: Uint8Array): void {
		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			this.ws.send(bytes);
		}
	}

	get connected(): boolean {
		return this._connected;
	}

	get protocolMismatch(): boolean {
		return this._protocolMismatch;
	}

	/** Issue an RPC. Returns a promise resolving to the server's result. */
	call<T = unknown>(op: string, payload: Record<string, unknown> = {}): Promise<T> {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			return Promise.reject(new Error('control socket not connected'));
		}
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
			// `session` rides every request at the top level (a sibling of op/payload) — the manager
			// scopes its command history's undo/redo by it. Harmless on read-only ops.
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
