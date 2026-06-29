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
 * (goofi/bridge/control.py PROTOCOL_VERSION) and the client asserts a match —
 * turning silent skew into an explicit "reload required". Bump BOTH sides
 * together whenever the wire shape or reconciliation rules change. */
export const PROTOCOL_VERSION = 1;

/** Whether a backend-reported protocol version is compatible with this build.
 * Strict equality: an absent/non-numeric version means a backend older than this
 * field, which is itself a skew worth flagging. */
export function isProtocolCompatible(remote: unknown): boolean {
	return remote === PROTOCOL_VERSION;
}

export interface NodeTypeInfo {
	type: string;
	category: string;
	doc: string;
	input_slots: Record<string, string>;
	output_slots: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
}

/** A node's self-reported execution telemetry (mirrors the backend NODE_STATS
 * payload). `updates_per_second` is the tick cadence; `mean_process_ms` the mean
 * `process()` wall-time over the window; `total_ticks` the lifetime count. */
export interface NodeStats {
	updates_per_second: number;
	mean_process_ms: number;
	total_ticks: number;
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
	output_slots: Record<string, string>;
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
	/** Distinct crash state (vs a code `error`): set when the node's OS process
	 * died and the manager is auto-restarting it; cleared on the respawned node's
	 * first healthy state push. `restarts` is the cumulative restart count. These
	 * are transient runtime UI state, populated by the `node_crashed` event — not
	 * part of the persisted snapshot. */
	crashed?: boolean;
	restarts?: number;
	crashExit?: number | null;
	/** Rolling execution telemetry the node pushes on the status plane (~1 Hz):
	 * update rate + mean `process()` duration over the last ~10 ticks. Absent until
	 * the node's first NODE_STATS push; populated by the `node_stats` event and
	 * present in the snapshot for a freshly-connected client. Runtime UI state. */
	stats?: NodeStats | null;
	/** Peer-to-peer SSE log endpoint (`http://127.0.0.1:<port>/<node>`) the node
	 * advertises via STATE_UPDATE. Null/absent until its first state push, or
	 * when capture is off (headless). The Console subscribes to it directly. */
	log_endpoint?: string | null;
	/** Set only on a *virtual* node synthesized for a sub-patch instance (see
	 * `graph.nodeById`). Lets the node layers — selection, inspector, drag —
	 * treat a sub-patch like a node while the inspector renders sharing controls
	 * instead of param groups. Absent/null for real nodes. */
	subpatch?: SubpatchMeta | null;
}

/** Marks a virtual node as standing in for a sub-patch instance. Only the id
 * rides along — live sharing state (kind, def_id, siblings, member count) is
 * recomputed from the `instances` map so it never goes stale. */
export interface SubpatchMeta {
	instId: string;
	/** Strict-mirror shared instance (vs a unique/inline sub-patch). */
	shared: boolean;
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
	pos?: [number, number];
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
			params: {}
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
	/** The instance's stable uid (also its key in the instances map). */
	uid: string;
	/** Display label, e.g. "subpatch0" — separate from the uid key. */
	name: string;
	kind: string;
	/** Definition id when shared (strict-mirror sibling), null/absent when unique. */
	def_id?: string | null;
	/** Parent instance uid (the nesting tree edge); null at the top level. */
	parent: string | null;
	/** boundary handle name -> port (dtype RESOLVED chain-to-leaf server-side) */
	interface: Record<string, SubPatchPort>;
	pos: [number, number];
	/** template-local name -> { member uid, whether the member is itself a nested
	 * instance }. Inverted from the backend's uid->local so the editor can split
	 * direct children into plain nodes vs nested collapsed instances. */
	members: Record<string, { uid: string; is_instance: boolean }>;
	/** External ports computed from WIRED boundaries: boundary id -> dtype. Mirror the
	 * collapsed group node's input/output slots (a pure passthrough). */
	slots: { input: Record<string, string>; output: Record<string, string> };
	/** Other instance uids in this instance's strict-mirror family (shared def); []. */
	siblings: string[];
	/** First errored DESCENDANT across the whole subtree (recursion-correct), or null. */
	error: string | null;
	/** Per-output-boundary view state persisted in the .gfi patch (round-trips), keyed
	 * by boundary id — same shape as a node's `viewers`. */
	viewers: Record<string, { collapsed?: boolean; kind?: string; settings?: Record<string, unknown> }>;
	member_count: number;
}

/** Canonical string key for a link — its four slot endpoints. Stable and
 * unique per link, usable as a map key or the SvelteFlow edge id. */
export function linkKey(l: LinkInfo): string {
	return `${l.node_out}.${l.slot_out}→${l.node_in}.${l.slot_in}`;
}

/** Structural equality of two links (all four endpoints match). */
export function sameLink(a: LinkInfo, b: LinkInfo): boolean {
	return (
		a.node_out === b.node_out &&
		a.node_in === b.node_in &&
		a.slot_out === b.slot_out &&
		a.slot_in === b.slot_in
	);
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
	nodes: NodeInstanceInfo[];
	links: LinkInfo[];
	/** Sub-patch instances keyed by instance id. */
	instances?: Record<string, InstanceInfo>;
	save_path: string | null;
	unsaved_changes: boolean;
	/** Opaque frontend workspace-layout blob restored from the .gfi patch.
	 * Null/absent when the patch carries no layout (older patch, or blank
	 * start) — a fresh session then falls back to the default layout. */
	layout?: unknown;
}

export type ControlEvent =
	| { event: 'hello'; payload: GraphSnapshot }
	| { event: 'node_added'; payload: NodeInstanceInfo }
	| { event: 'node_removed'; payload: { node: string } }
	| { event: 'node_moved'; payload: { node: string; pos: [number, number] } }
	| { event: 'link_added'; payload: LinkInfo }
	| { event: 'link_removed'; payload: LinkInfo }
	| {
			event: 'state_update';
			payload: {
				node: string;
				params: Record<string, Record<string, ParamDescriptor>>;
				output_subscribers: Record<string, number>;
				log_endpoint?: string | null;
			};
	  }
	| { event: 'error'; payload: { node: string; error: string | null } }
	| { event: 'node_crashed'; payload: { node: string; exitcode: number | null; restarts: number } }
	| { event: 'node_stats'; payload: { node: string; stats: NodeStats } }
	| { event: 'unsaved_changes'; payload: { unsaved_changes: boolean } }
	| { event: 'save_path_changed'; payload: { save_path: string | null } }
	| { event: 'graph_replaced'; payload: GraphSnapshot }
	| { event: 'subpatch_changed'; payload: GraphSnapshot }
	| { event: 'boundary_moved'; payload: { inst_id: string; bnd_id: string; pos: [number, number] } }
	| { event: 'node_renamed'; payload: { node: string; name: string } }
	| { event: 'layout'; payload: { layout: unknown } }
	| { event: 'manager_shutdown'; payload: Record<string, never> };

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
}

export class ControlClient implements Control {
	private ws: WebSocket | null = null;
	private url: string;
	private nextId = 1;
	private pending = new Map<number, Pending>();
	private handlers = new Set<EventHandler>();
	private connectListeners = new Set<(connected: boolean) => void>();
	private protocolListeners = new Set<(mismatch: boolean) => void>();
	private _connected = false;
	private _protocolMismatch = false;
	private retryMs = 250;
	private closedByUser = false;

	constructor(url?: string) {
		const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
		this.url = url ?? `${proto}//${location.host}/control`;
	}

	connect(): void {
		this.closedByUser = false;
		this._open();
	}

	private _open(): void {
		if (this.ws) return;
		const ws = new WebSocket(this.url);
		ws.binaryType = 'arraybuffer';
		this.ws = ws;

		ws.addEventListener('open', () => {
			this.retryMs = 250;
			this._setConnected(true);
		});
		ws.addEventListener('message', (e) => this._onMessage(e));
		ws.addEventListener('close', () => this._onClose());
		ws.addEventListener('error', () => {
			try {
				ws.close();
			} catch {
				/* noop */
			}
		});
	}

	private _onMessage(e: MessageEvent): void {
		if (typeof e.data !== 'string') return;
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
		if (this.closedByUser) return;
		const delay = this.retryMs;
		this.retryMs = Math.min(this.retryMs * 2, 5000);
		setTimeout(() => this._open(), delay);
	}

	private _setConnected(v: boolean): void {
		if (this._connected === v) return;
		this._connected = v;
		for (const h of this.connectListeners) h(v);
	}

	close(): void {
		this.closedByUser = true;
		this.ws?.close();
		this.ws = null;
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
			this.ws!.send(JSON.stringify({ id, op, payload }));
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
