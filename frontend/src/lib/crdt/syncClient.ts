/**
 * The client-side sync driver: binds a `Y.Doc` to a `Control`'s binary sync channel so the
 * browser replica converges with the manager's authoritative doc and tracks live deltas.
 *
 * Pure of Svelte — it wires callbacks and a doc, so it unit-tests against a `FakeControl`.
 * The reactive `.svelte.ts` layer wraps a `SyncClient` and re-exposes the doc as runes.
 */
import * as Y from 'yjs';
import type { Control } from '$lib/api/control';
import { decodeSyncMsg, onSync, syncHello } from './syncProtocol';

/** Origin tag stamped on transactions produced by applying a REMOTE (synced) update, so
 * doc observers can distinguish the manager's changes from this client's own local edits. */
export const REMOTE_ORIGIN = 'goofi:remote';

export class SyncClient {
	readonly doc: Y.Doc;
	private control: Control;
	private unsub: (() => void) | null = null;

	constructor(control: Control, doc: Y.Doc = new Y.Doc()) {
		this.control = control;
		this.doc = doc;
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
