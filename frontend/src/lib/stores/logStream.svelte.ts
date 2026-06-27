/**
 * Peer-to-peer log subscription manager.
 *
 * Each node process exposes its own SSE log endpoint (advertised by the backend
 * as `node.log_endpoint`). This manager opens a native `EventSource` straight to
 * each node — the manager/bridge never relays the bytes — and feeds the records
 * into the console store.
 *
 * Subscription is **lazy**: a Console panel declares the nodes it needs (`'all'`
 * while unfiltered, a single node when drag-filtered) and releases them on
 * unmount. We open connections only for the union of what the open panels need
 * and only for nodes whose endpoint is known, and we close them as soon as a
 * node is no longer needed, its endpoint changes, or it is removed. With no
 * Console open, nothing is connected.
 *
 * A single reactive effect reconciles the live `EventSource` set against the
 * needed nodes and the graph's advertised endpoints.
 */
import { untrack } from 'svelte';
import { graph } from './graph.svelte';
import { consoleStore, type LogRecord } from './console.svelte';

type Need = 'all' | string[];

class LogStream {
	private needs = $state<Record<string, Need>>({});
	private sources = new Map<string, { es: EventSource; endpoint: string }>();
	private started = false;

	/** A panel declares which nodes it needs streamed. Mutating `needs` is
	 * untracked so a *caller's* effect (which only means to depend on its own
	 * filter) doesn't pick up `needs` as a dependency and self-loop — the
	 * dedicated reconcile effect below is the sole tracked reader of `needs`. */
	setNeeds(panelId: string, need: Need): void {
		untrack(() => {
			this.needs = { ...this.needs, [panelId]: need };
		});
		this.ensureStarted();
	}

	/** A panel stops needing any streams (unmount / no longer a console). */
	release(panelId: string): void {
		untrack(() => {
			if (!(panelId in this.needs)) return;
			const { [panelId]: _drop, ...rest } = this.needs;
			this.needs = rest;
		});
	}

	private neededNames(names: Iterable<string>): Set<string> {
		const out = new Set<string>();
		let all = false;
		for (const n of Object.values(this.needs)) {
			if (n === 'all') {
				all = true;
			} else {
				for (const name of n) out.add(name);
			}
		}
		if (all) for (const name of names) out.add(name);
		return out;
	}

	private ensureStarted(): void {
		if (this.started) return;
		this.started = true;
		// A standalone root so the reconcile effect lives for the app's lifetime,
		// reacting to needs + the graph's advertised endpoints.
		$effect.root(() => {
			$effect(() => this.reconcile());
		});
	}

	private reconcile(): void {
		const nodes = graph().nodes;
		// Key endpoints by the stable uid — the same identity the need set carries
		// (a Console's bound node is its uid) and that the console store filters /
		// error-correlates by. Keying by the mutable display name would make a
		// node-filtered Console open no stream at all.
		const endpoints = new Map<string, string>();
		for (const node of nodes) {
			if (node.log_endpoint) endpoints.set(node.uid, node.log_endpoint);
		}
		const need = this.neededNames(endpoints.keys());

		// Close what's no longer needed or whose endpoint moved.
		for (const [uid, src] of [...this.sources]) {
			if (!need.has(uid) || endpoints.get(uid) !== src.endpoint) {
				this.closeOne(uid);
			}
		}
		// Open what's needed and known but not yet connected.
		for (const uid of need) {
			const ep = endpoints.get(uid);
			if (ep && !this.sources.has(uid)) this.open(uid, ep);
		}
	}

	private open(uid: string, endpoint: string): void {
		if (typeof EventSource === 'undefined') return;
		let es: EventSource;
		try {
			es = new EventSource(endpoint);
		} catch {
			return;
		}
		es.onmessage = (e: MessageEvent) => {
			try {
				const rec = JSON.parse(e.data) as LogRecord;
				// SSE frames identify the source by the node-process `node_id` (a
				// third identity the rest of the frontend never sees). Re-stamp with
				// the owning uid so entries match the uid-keyed filters + mirrored
				// errors in the console store.
				rec.node = uid;
				consoleStore().ingest(rec);
			} catch {
				/* ignore malformed frame */
			}
		};
		// EventSource auto-reconnects on error; the backend resumes via Last-Event-ID.
		this.sources.set(uid, { es, endpoint });
	}

	private closeOne(uid: string): void {
		const src = this.sources.get(uid);
		if (!src) return;
		try {
			src.es.close();
		} catch {
			/* noop */
		}
		this.sources.delete(uid);
	}
}

let _stream: LogStream | null = null;
export function logStream(): LogStream {
	if (!_stream) _stream = new LogStream();
	return _stream;
}
