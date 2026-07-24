/**
 * The client-side sync driver: binds a `Y.Doc` to a `Control`'s binary sync channel so the
 * browser replica converges with the manager's authoritative doc and tracks live deltas.
 *
 * Pure of Svelte — it wires callbacks and a doc, so it unit-tests against a `FakeControl`.
 * The reactive `.svelte.ts` layer wraps a `SyncClient` and re-exposes the doc as runes.
 */
import * as Y from 'yjs';
import type { Control } from '$lib/api/control';
import { decodeSyncMsg, encodeSyncMsg, onSync, syncHello } from './syncProtocol';

/** Origin tag stamped on transactions produced by applying a REMOTE (synced) update, so
 * doc observers can distinguish the manager's changes from this client's own local edits. */
export const REMOTE_ORIGIN = 'goofi:remote';

export class SyncClient {
	/** The replica doc. Swapped by `reset()` on a fresh backend session, so read it via the getter
	 * (never cache the reference across a reset). */
	private _doc: Y.Doc;
	get doc(): Y.Doc {
		return this._doc;
	}
	private control: Control;
	private unsub: (() => void) | null = null;
	/** The graph-store's reactive doc-change observer — re-attached to the fresh doc on `reset()`. */
	private docObserver: ((txn: Y.Transaction) => void) | null = null;

	constructor(control: Control, doc: Y.Doc = new Y.Doc()) {
		this.control = control;
		this._doc = doc;
	}

	/** Register the store's reactive doc-change callback (fired on every doc transaction). Survives
	 * `reset()` — re-attached to the fresh doc. Replaces attaching `doc.on('afterTransaction', …)`
	 * directly, which would be left dangling on the destroyed doc after a reset. */
	onDocChange(fn: (txn: Y.Transaction) => void): void {
		if (this.docObserver) this._doc.off('afterTransaction', this.docObserver);
		this.docObserver = fn;
		this._doc.on('afterTransaction', fn);
	}

	/** Replace the replica with a fresh, EMPTY doc — call when the manager reports a NEW backend
	 * session (changed instance_id). The previous session's content must NOT survive: a fresh engine
	 * mints uids from 1 again, so the stale doc collides on reused uids, and the reconnect handshake
	 * would (a) push stale param/pos/expr leaves into the new session when answering the server's SV
	 * and (b) merge new-session content into the stale doc — both silent corruption. A fresh empty
	 * doc answers the server's SV with nothing (no stale push) and pulls the new session cleanly.
	 * MUST run synchronously in the hello handler, before the client answers the server's binary SV. */
	reset(): void {
		const wasRunning = this.unsub !== null;
		this.stop();
		if (this.docObserver) this._doc.off('afterTransaction', this.docObserver);
		this._doc.destroy();
		this._doc = new Y.Doc();
		if (this.docObserver) this._doc.on('afterTransaction', this.docObserver);
		if (wasRunning) this.start();
	}



	/** Begin syncing: subscribe to inbound frames, and advertise our state vector on connect
	 * AND on every reconnect so we re-pull anything the manager changed while we were
	 * disconnected. Advertising only once at mount would leave the replica silently diverged
	 * after any WS drop (the manager re-sends only its own SV on reconnect, which a reader
	 * answers with an empty diff — so the client must be the one to re-advertise). Idempotent. */
	start(): void {
		if (this.unsub) return;
		const offFrame = this.control.onSyncFrame((bytes) => this.onFrame(bytes));
		// onConnect fires immediately with the current state (initial mount) and again on each
		// reconnect; each `true` re-advertises our SV, prompting the manager's diff.
		const offConn = this.control.onConnect((connected) => {
			if (connected) this.control.sendSync(syncHello(this.doc));
		});
		this.unsub = () => {
			offFrame();
			offConn();
		};
	}

	/** Stop syncing (unsubscribe). The doc is retained. */
	stop(): void {
		this.unsub?.();
		this.unsub = null;
	}

	/** Drive one inbound frame through the handshake, sending back any replies. Exposed for
	 * tests; normally called from the `onSyncFrame` subscription. */
	onFrame(bytes: Uint8Array): void {
		const msg = decodeSyncMsg(bytes);
		if (!msg) return;
		const replies = onSync(this.doc, msg, REMOTE_ORIGIN);
		for (const r of replies) this.control.sendSync(r);
	}
}
