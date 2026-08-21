/** The document driver: follows `doc_state` / `doc_patch` so the replica equals the manager's document. */
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
	/** Whether this replica has been seeded yet — until it flips, reads describe an empty replica. */
	get synced(): boolean {
		return this._version >= 0;
	}

	constructor(control: Control) {
		this.control = control;
	}

	onDocChange(fn: () => void): void {
		this.docObserver = fn;
	}

	/** Drop the replica to empty for a NEW backend session, whose engine remints uids from 1. */
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

	/** Apply one delta: a patch already held is skipped, one reaching past this version is refused. */
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
