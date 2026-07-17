/**
 * Reactive Svelte 5 view over the CRDT control-plane document. Wraps a {@link SyncClient}
 * (which keeps the browser `Y.Doc` converged with the manager) and re-exposes the doc's
 * node/param/link state as runes, so components read patch state from the CRDT rather than
 * from the hand-applied event stream (Phase 2, reader half).
 *
 * The testable logic (framing, handshake, doc reads) lives in the sibling pure modules —
 * this file is the rune glue, verified by `npm run check` + e2e.
 */
import * as Y from 'yjs';
import { getControl, type Control } from '$lib/api/control';
import { SyncClient } from './syncClient';
import { nodeViews, linkViews, type NodeView, type LinkView } from './graphDoc';

export class CrdtGraph {
	private sync: SyncClient;

	/** Node identity views, kept in sync with the doc. */
	nodes = $state<NodeView[]>([]);
	/** Links, kept in sync with the doc. */
	links = $state<LinkView[]>([]);

	constructor(control: Control = getControl()) {
		this.sync = new SyncClient(control);
		// Re-read the reactive snapshots whenever the doc changes (local or remote).
		this.sync.doc.on('afterTransaction', (txn: Y.Transaction) => {
			if (txn.changed.size > 0) this.refresh();
		});
	}

	/** The underlying Yjs document (for finer-grained reads by inspector widgets). */
	get doc(): Y.Doc {
		return this.sync.doc;
	}

	/** Begin syncing from the manager. Call once on mount. */
	start(): void {
		this.sync.start();
	}

	stop(): void {
		this.sync.stop();
	}

	private refresh(): void {
		this.nodes = nodeViews(this.sync.doc);
		this.links = linkViews(this.sync.doc);
	}
}

let _crdt: CrdtGraph | null = null;
/** Process-wide singleton reactive CRDT graph (one bridge ↔ one tab). */
export function crdtGraph(): CrdtGraph {
	if (!_crdt) {
		_crdt = new CrdtGraph();
		_crdt.start();
	}
	return _crdt;
}
