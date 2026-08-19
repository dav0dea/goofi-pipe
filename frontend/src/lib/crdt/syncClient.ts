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
	 * Apply one delta. A patch that does not follow this replica's version is REFUSED, not merged:
	 * the events arrive in order on one socket, so a gap means the client fell behind the broadcast
	 * ring — and the manager answers that by re-sending `doc_state`, which is what heals it. Merging
	 * anyway would leave a replica that looks healthy and is wrong.
	 *
	 * Exposed for tests; normally driven from the subscription in [`start`].
	 */
	applyPatch(from: number, to: number, patch: Record<string, unknown>): void {
		if (from !== this._version) {
			console.warn(`goofi: doc patch applies to v${from}, this replica is at v${this._version} — waiting for a fresh doc_state`);
			return;
		}
		applyMerge(this._doc, patch);
		this._version = to;
		this.docObserver?.();
	}
}
