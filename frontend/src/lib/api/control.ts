/**
 * Control-plane WebSocket client.
 *
 * Connects to `/control`, exposes a typed RPC method (`call`), and an
 * event subscription mechanism (`on`). Auto-reconnects with exponential
 * backoff; pending RPCs that were in-flight when the socket dropped are
 * rejected so callers can re-issue if appropriate.
 */
import type { ParamDescriptor } from '$lib/api/types';

export interface NodeTypeInfo {
	type: string;
	category: string;
	doc: string;
	input_slots: Record<string, string>;
	output_slots: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
}

export interface NodeInstanceInfo {
	name: string;
	type: string;
	category: string;
	doc: string;
	input_slots: Record<string, string>;
	output_slots: Record<string, string>;
	params: Record<string, Record<string, ParamDescriptor>>;
	pos: [number, number];
	/** Per-output-slot view state restored from the .gfi patch — empty when
	 * the node was just spawned, populated when a save was previously made. */
	viewers: Record<string, { collapsed?: boolean }>;
	error: string | null;
	/** Peer-to-peer SSE log endpoint (`http://127.0.0.1:<port>/<node>`) the node
	 * advertises via STATE_UPDATE. Null/absent until its first state push, or
	 * when capture is off (headless). The Console subscribes to it directly. */
	log_endpoint?: string | null;
}

export interface LinkInfo {
	node_out: string;
	node_in: string;
	slot_out: string;
	slot_in: string;
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
	/** Identifies the manager *process*. Changes when the backend is restarted,
	 * letting the graph store distinguish a fresh session (reset the layout)
	 * from a transient reconnect to the same one (keep the current layout). */
	instance_id: string;
	nodes: NodeInstanceInfo[];
	links: LinkInfo[];
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
	| { event: 'node_removed'; payload: { name: string } }
	| { event: 'node_moved'; payload: { name: string; pos: [number, number] } }
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
	| { event: 'unsaved_changes'; payload: { unsaved_changes: boolean } }
	| { event: 'save_path_changed'; payload: { save_path: string | null } }
	| { event: 'graph_replaced'; payload: GraphSnapshot }
	| { event: 'layout'; payload: { layout: unknown } }
	| { event: 'manager_shutdown'; payload: Record<string, never> };

type EventHandler = (ev: ControlEvent) => void;

type Pending = {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
};

export class ControlClient {
	private ws: WebSocket | null = null;
	private url: string;
	private nextId = 1;
	private pending = new Map<number, Pending>();
	private handlers = new Set<EventHandler>();
	private connectListeners = new Set<(connected: boolean) => void>();
	private _connected = false;
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
			for (const h of this.handlers) {
				try {
					h(msg as ControlEvent);
				} catch (err) {
					console.error('control handler crashed', err);
				}
			}
		}
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

	get connected(): boolean {
		return this._connected;
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
