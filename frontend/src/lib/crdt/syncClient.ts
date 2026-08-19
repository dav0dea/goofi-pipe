/**
 * The client-side document driver: follows the manager's `doc_state` / `doc_patch` events so the
 * browser replica stays equal to the manager's control-plane document.
 *
 * Pure of Svelte — it wires callbacks and a plain object, so it unit-tests against a `FakeControl`.
 * The reactive `.svelte.ts` layer wraps a `SyncClient` and re-exposes the document as runes.
 *
 * The replica is READ-ONLY. Every mutation is an RPC the manager applies, and the delta comes back
 * here — so there is nothing to merge, and no writer but this one.
 */
import type { Control } from '$lib/api/control';
import { applyMerge } from './mergePatch';
import { emptyDoc, type Doc } from './graphDoc';

export class SyncClient {
	private _doc: Doc = emptyDoc();
	get doc(): Doc {
		return this._doc;
	}
	private control: Control;
	private unsub: (() => void) | null = null;
	private docObserver: (() => void) | null = null;
	/** The version `_doc` is at, or `-1` before the first `doc_state`. */
	private _version = -1;
	get version(): number {
		return this._version;
	}
	/** Whether this replica has been seeded yet. Until it flips, reads describe an empty replica
	 * rather than the graph — the automation façade exposes it as `query.docSynced` so a driver can
	 * tell "empty" from "not yet delivered". */
	get synced(): boolean {
		return this._version >= 0;
	}

	constructor(control: Control) {
		this.control = control;
	}

	/** Register the store's reactive change callback, fired after every applied change. */
	onDocChange(fn: () => void): void {
		this.docObserver = fn;
	}

	/**
	 * Drop the replica back to empty — call when the manager reports a NEW backend session (changed
	 * instance_id). The previous session's content must NOT survive: a fresh engine mints uids from
	 * 1 again, so a stale document would collide on reused uids and its leaves would read as the new
	 * session's. The manager sends `doc_state` on every connection, so the empty replica is filled
	 * again without asking for anything.
	 */
	reset(): void {
		this._doc = emptyDoc();
		this._version = -1;
		this.docObserver?.();
	}

	/** Begin following the document. Idempotent. */
	start(): void {
		if (this.unsub) return;
		this.unsub = this.control.on((ev) => {
			if (ev.event === 'doc_state') {
				this._doc = ev.payload.doc;
				this._version = ev.payload.v;
				this.docObserver?.();
			} else if (ev.event === 'doc_patch') {
				this.applyPatch(ev.payload.from, ev.payload.v, ev.payload.patch);
			}
		});
	}

	/** Stop following. The document is retained. */
	stop(): void {
		this.unsub?.();
		this.unsub = null;
	}

	/**
	 * Apply one delta. A version that does not match means one of two different things.
	 *
	 * A patch whose RESULT this replica already holds is STALE, and skipping it is unremarkable:
	 * the manager subscribes a socket before it snapshots the document, so a peer's edit landing in
	 * that window is broadcast and then included in the snapshot as well. Re-delivery is the price
	 * of never losing one, and this is where it is paid.
	 *
	 * A patch reaching FORWARD of this replica is a GAP — the client fell behind the broadcast ring
	 * — and it is refused, not merged. Applying onto the wrong base would leave a replica that
	 * looks healthy and is wrong. The manager answers that same lag by re-sending the whole
	 * document, which is what heals it.
	 *
	 * Exposed for tests; normally driven from the subscription in [`start`].
	 */
	applyPatch(from: number, to: number, patch: Record<string, unknown>): void {
		if (to <= this._version) return; // stale: the seed already carries it
		if (from !== this._version) {
			console.warn(
				`goofi: doc patch spans v${from}→v${to} but this replica is at v${this._version} — a delta was lost; waiting for a fresh doc_state`
			);
			return;
		}
		applyMerge(this._doc, patch);
		this._version = to;
		this.docObserver?.();
	}
}
