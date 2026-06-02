/**
 * Console log store — bounded, virtualizable transcripts of node stdout/stderr.
 *
 * Fed by the per-node SSE streams (see `$lib/stores/logStream`) plus processing
 * errors mirrored as stderr (see the graph store's `error` handler). Designed to
 * stay smooth at ~100k entries.
 *
 * Each *view* (the unfiltered "everything" view, and one per active filter) owns
 * an independent **coalesced** stream:
 *
 * - storage is a fixed **circular buffer** (`Ring`): O(1) append and O(1) evict
 *   of the oldest, no `Array.shift`.
 * - consecutive identical lines (same node + stream + text) coalesce into one
 *   entry whose `count` ticks up — Chrome-devtools style. Coalescing is computed
 *   *per view*, so a node-filtered view groups that node's consecutive errors
 *   even when another node's output interleaved them in the global stream.
 * - within a view, `uid` increments only on append (not on coalesce), giving each
 *   entry a stable identity (the `{#each}` key) that a coalesced repeat preserves.
 * - a `version` rune bumps at most once per animation frame, so a burst of
 *   appends triggers one re-render, not one per line. Entries are plain objects
 *   (the panel renders shallow copies of the visible window so an in-place
 *   `count` bump re-renders the keyed row).
 * - replay is deduped by a per-node monotonic `seq`, so an SSE reconnect or a
 *   panel re-subscribe can't duplicate lines it already holds.
 *
 * The unfiltered view is permanent so logs/errors accumulate even with no
 * console open; a filtered view is built on demand from the unfiltered view's
 * history (re-coalesced for the filter) and then maintained incrementally. The
 * only place lines are dropped is the per-view cap (oldest first).
 */

export type LogStream = 'stdout' | 'stderr';

/** One ingested log record off the wire (or a mirrored error). */
export interface LogRecord {
	node: string;
	stream: LogStream;
	text: string;
	/** Per-node monotonic sequence from the backend buffer (replay dedup). */
	seq: number;
	ts: number;
}

export interface ConsoleEntry {
	/** Monotonic id within its view, stable across eviction — the `{#each}` key. */
	uid: number;
	node: string;
	stream: LogStream;
	text: string;
	/** Consecutive-duplicate count (Chrome-style ×N). */
	count: number;
	ts: number;
	/** Logical line count (newlines + 1) — the panel's row-height estimate before
	 * a row is measured. */
	lines: number;
}

function countLines(text: string): number {
	let n = 1;
	for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n++;
	return n;
}

/** A read handle over one view — what a Console panel renders. */
export interface ConsoleView {
	/** Filter signature; pass to `releaseView` when done. */
	sig: string;
	total(): number;
	/** Entry at filtered position `i` (0 = oldest). */
	get(i: number): ConsoleEntry;
}

const MAX_ENTRIES = 100_000;

/** Fixed-capacity circular buffer: O(1) push / evict-oldest / indexed read. */
class Ring<T> {
	private buf: (T | undefined)[];
	private head = 0;
	count = 0;

	constructor(public cap: number) {
		this.buf = new Array(cap);
	}

	get(i: number): T {
		return this.buf[(this.head + i) % this.cap] as T;
	}

	last(): T | undefined {
		return this.count ? this.get(this.count - 1) : undefined;
	}

	push(v: T): void {
		if (this.count < this.cap) {
			this.buf[(this.head + this.count) % this.cap] = v;
			this.count++;
		} else {
			this.buf[this.head] = v;
			this.head = (this.head + 1) % this.cap;
		}
	}

	clear(): void {
		this.buf = new Array(this.cap);
		this.head = 0;
		this.count = 0;
	}
}

/** One coalesced stream for a filter. */
interface View {
	sig: string;
	node: string | null;
	stdout: boolean;
	stderr: boolean;
	ring: Ring<ConsoleEntry>;
	/** Next uid; increments on append only (not coalesce) → consecutive uids. */
	uidSeq: number;
	refs: number;
	/** The unfiltered view is permanent (accumulates with no console open). */
	permanent: boolean;
}

const ALL_SIG = '*|1|1';

function viewSig(node: string | null, stdout: boolean, stderr: boolean): string {
	return `${node ?? '*'}|${stdout ? 1 : 0}|${stderr ? 1 : 0}`;
}

export class ConsoleStore {
	private views = new Map<string, View>();
	private dedup = new Map<string, number>();

	/** Bumped (rAF-coalesced) whenever any view's visible data changes — drives
	 * the panel's row re-render (incl. ×count badges). */
	version = $state(0);
	/** Bumped only when the entry *set* changes (a new entry pushed / cleared),
	 * NOT on a coalesce count-bump. The panel keys its (more expensive) row-height
	 * layout off this so a stream of coalescing duplicates doesn't rebuild it. */
	layoutVersion = $state(0);
	private bumpScheduled = false;

	constructor() {
		// The unfiltered "everything" view is permanent.
		this.views.set(ALL_SIG, this.makeView(ALL_SIG, null, true, true, true));
	}

	private makeView(
		sig: string,
		node: string | null,
		stdout: boolean,
		stderr: boolean,
		permanent: boolean
	): View {
		return { sig, node, stdout, stderr, ring: new Ring<ConsoleEntry>(MAX_ENTRIES), uidSeq: 0, refs: 0, permanent };
	}

	private matches(node: string, stream: LogStream, v: View): boolean {
		if (v.node !== null && node !== v.node) return false;
		return stream === 'stdout' ? v.stdout : v.stderr;
	}

	/** Append one line into a view, coalescing a consecutive identical line.
	 * Returns true if a new entry was pushed (vs. coalesced into the last). */
	private appendTo(v: View, node: string, stream: LogStream, text: string, ts: number, count: number): boolean {
		const last = v.ring.last();
		if (last && last.node === node && last.stream === stream && last.text === text) {
			last.count += count;
			last.ts = ts;
			return false;
		}
		v.ring.push({ uid: v.uidSeq++, node, stream, text, count, ts, lines: countLines(text) });
		return true;
	}

	private feed(node: string, stream: LogStream, text: string, ts: number): void {
		let pushed = false;
		for (const v of this.views.values()) {
			if (this.matches(node, stream, v) && this.appendTo(v, node, stream, text, ts, 1)) pushed = true;
		}
		if (pushed) this.layoutVersion += 1;
		this.scheduleBump();
	}

	/** Ingest one SSE record. Synchronous on the data; re-render is rAF-batched. */
	ingest(rec: LogRecord): void {
		// Replay dedup: drop anything at/below the last seq seen for this node.
		const seen = this.dedup.get(rec.node);
		if (seen !== undefined && rec.seq <= seen) return;
		this.dedup.set(rec.node, rec.seq);
		this.feed(rec.node, rec.stream, rec.text, rec.ts);
	}

	/** Mirror a processing-error traceback as a stderr entry (bypasses seq dedup —
	 * it comes off the control plane, not the SSE stream). Repeats coalesce. */
	ingestError(node: string, text: string, ts: number): void {
		this.feed(node, 'stderr', text, ts);
	}

	/** Forget a node's dedup cursor (it was removed — a future re-add resets). */
	forgetNodeDedup(node: string): void {
		this.dedup.delete(node);
	}

	clear(): void {
		// Keep uidSeq monotonic across clears so uids never get reused mid-session.
		for (const v of this.views.values()) v.ring.clear();
		this.dedup.clear();
		this.layoutVersion += 1;
		this.scheduleBump();
	}

	/** Acquire a render handle for a filter (ref-counted). Release with the sig. */
	acquireView(node: string | null, stdout: boolean, stderr: boolean): ConsoleView {
		const sig = node === null && stdout && stderr ? ALL_SIG : viewSig(node, stdout, stderr);
		let v = this.views.get(sig);
		if (!v) {
			v = this.makeView(sig, node, stdout, stderr, false);
			// Seed history from the unfiltered view, re-coalescing for this filter
			// (so the filtered node's groups are correct from the first frame).
			const all = this.views.get(ALL_SIG);
			if (all) {
				for (let i = 0; i < all.ring.count; i++) {
					const e = all.ring.get(i);
					if (this.matches(e.node, e.stream, v)) this.appendTo(v, e.node, e.stream, e.text, e.ts, e.count);
				}
			}
			this.views.set(sig, v);
		}
		v.refs += 1;
		const ring = v.ring;
		return {
			sig,
			total: () => ring.count,
			get: (i) => ring.get(i)
		};
	}

	releaseView(sig: string): void {
		const v = this.views.get(sig);
		if (!v || v.permanent) return;
		v.refs -= 1;
		if (v.refs <= 0) this.views.delete(sig);
	}

	private scheduleBump(): void {
		if (this.bumpScheduled) return;
		this.bumpScheduled = true;
		const run = () => {
			this.bumpScheduled = false;
			this.version += 1;
		};
		if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
		else queueMicrotask(run);
	}
}

let _store: ConsoleStore | null = null;
export function consoleStore(): ConsoleStore {
	if (!_store) _store = new ConsoleStore();
	return _store;
}
