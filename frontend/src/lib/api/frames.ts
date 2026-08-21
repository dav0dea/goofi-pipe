/** The viewer registry and display-rate frame delivery: ONE rAF flush per tick, most-starved
 * slot first, with a per-frame time budget. */
import { closeStream, openStream, sendSpecs, setFrameSink } from './data';
import { paintDelay } from './paintCap';
import { perfStats } from './perfStats.svelte';
import { RateMeter } from './rateMeter';
import type { DataFrame } from '$lib/codec/decode';
import type { ViewSpec } from '$lib/viewers/capacity';
import { streamKey } from './streamKey';

type FrameCallback = (frame: DataFrame) => void;

/** One viewer bound to a stream. A null `spec` contributes nothing to the reduction, which an
 * empty LIST does not: that means full resolution. */
interface BoundViewer {
	cb: FrameCallback;
	spec: ViewSpec | null;
}

interface Slot {
	/** Every viewer bound to this stream, by its own stable token. THE registry: nothing else
	 *  counts viewers, and nothing else decides what the backend is asked for. */
	viewers: Map<string, BoundViewer>;
	pending: DataFrame | null;
	current: DataFrame | null;
	/** When this slot last delivered, for most-starved-first fairness. */
	lastFlush: number;
	/** This stream's own coalescing rate — per slot, because a drop belongs to the stream that
	 * overwrote a frame. */
	drops: RateMeter;
}

const slots = new Map<string, Slot>();
const dirty = new Set<Slot>();
const FRAME_BUDGET_MS = 8;

const nowMs =
	typeof performance !== 'undefined' && typeof performance.now === 'function'
		? (): number => performance.now()
		: (): number => Date.now();

const scheduleFlush =
	typeof requestAnimationFrame === 'function'
		? (fn: () => void): number => requestAnimationFrame(fn)
		: (fn: () => void): number => setTimeout(fn, 16) as unknown as number;

let scheduled = false;
/** When the last flush started — the paint cap's reference point. */
let lastFlushStart = -Infinity;
function requestFlush(): void {
	if (scheduled) return;
	scheduled = true;
	// Inside the cap's cooldown, hold the frames on a TIMER: an rAF would fire at display rate
	// just to decide "not yet". The rAF after it aligns the paint to the next vsync.
	const wait = paintDelay(lastFlushStart, nowMs());
	const arm = (): void => {
		scheduleFlush(() => {
			scheduled = false;
			flush();
		});
	};
	if (wait > 0) setTimeout(arm, wait);
	else arm();
}

function flush(): void {
	const start = nowMs();
	lastFlushStart = start;
	// Most-starved slot first, so the budget can never permanently defer a slot.
	const queue = [...dirty].sort((a, b) => a.lastFlush - b.lastFlush);
	let painted = 0;
	for (const s of queue) {
		// Always deliver at least one slot; after that, stop once over budget.
		if (painted > 0 && nowMs() - start > FRAME_BUDGET_MS) break;
		const frame = s.pending;
		dirty.delete(s);
		if (!frame) continue;
		s.pending = null;
		s.current = frame;
		s.lastFlush = start;
		painted++;
		for (const { cb } of [...s.viewers.values()]) {
			try {
				cb(frame);
			} catch (err) {
				console.error('frame consumer crashed', err);
			}
		}
	}
	// ONE paint per flush, not one per slot: that is the quantity the cap bounds and the HUD names.
	if (painted > 0) perfStats().delivered();
	if (dirty.size > 0) requestFlush();
}


/** What the backend has been told to serve for a stream; absent means no stream is open. */
const synced = new Map<string, string>();
const reconciling = new Set<string>();

/** Bring the backend in line with the registry for one stream, read once the tick has SETTLED. */
function reconcile(node: string, slot: string, k: string): void {
	const s = slots.get(k);
	const want = s && s.viewers.size > 0 ? JSON.stringify(demand(s)) : null;
	const have = synced.get(k) ?? null;
	if (want === have) return;

	if (want === null) {
		closeStream(node, slot);
		synced.delete(k);
		if (s) dirty.delete(s);
		slots.delete(k);
		return;
	}
	if (have === null) openStream(node, slot);
	sendSpecs(node, slot, JSON.parse(want) as ViewSpec[]);
	synced.set(k, want);
}

/** What this page needs a stream reduced to: every bound viewer's constraint, DISTINCT ones only
 * — the bridge folds richest-per-dim, so a repeat would renegotiate nothing. */
function demand(s: Slot): ViewSpec[] {
	const seen = new Set<string>();
	const out: ViewSpec[] = [];
	for (const { spec } of s.viewers.values()) {
		if (!spec) continue;
		const sig = JSON.stringify(spec);
		if (seen.has(sig)) continue;
		seen.add(sig);
		out.push(spec);
	}
	return out;
}

function scheduleReconcile(node: string, slot: string, k: string): void {
	if (reconciling.has(k)) return;
	reconciling.add(k);
	queueMicrotask(() => {
		reconciling.delete(k);
		reconcile(node, slot, k);
	});
}

function ensureSlot(k: string): Slot {
	let s = slots.get(k);
	if (!s) {
		s = {
			viewers: new Map(),
			pending: null,
			current: null,
			lastFlush: 0,
			drops: new RateMeter(nowMs())
		};
		slots.set(k, s);
	}
	return s;
}

/** Bind a viewer to a (node, slot) stream under `token`, so several viewers of one slot collect
 * rather than evict. Re-binding with a changed `spec` reports a resize or a kind switch. */
export function bindViewer(
	node: string,
	slot: string,
	token: string,
	spec: ViewSpec | null,
	cb: FrameCallback
): () => void {
	const k = streamKey(node, slot);
	const s = ensureSlot(k);
	s.viewers.set(token, { cb, spec });
	scheduleReconcile(node, slot, k);
	// An open stream sends nothing new for a joiner, so replay the current frame to it alone —
	// re-marking the slot dirty would repaint every settled viewer.
	if (s.current) {
		try {
			cb(s.current);
		} catch (err) {
			console.error('frame consumer crashed', err);
		}
	}
	return () => {
		const cur = slots.get(k);
		if (cur?.viewers.get(token)?.cb !== cb) return; // already replaced by a later bind
		cur.viewers.delete(token);
		scheduleReconcile(node, slot, k);
	};
}

/** Everything the worker decodes lands here, latest-wins, and is painted on the next flush. */
setFrameSink((node, slot, frame) => {
	const s = slots.get(streamKey(node, slot));
	if (!s) return;
	// A pending frame overwritten before it painted is a drop, charged to THIS stream.
	if (s.pending !== null) s.drops.dropped();
	s.pending = frame;
	dirty.add(s);
	requestFlush();
});

/** Coalesced-frame rate for ONE stream. Null when nothing is subscribed: absent is not zero. */
export function dropRate(node: string, slot: string): number | null {
	const s = slots.get(streamKey(node, slot));
	if (!s) return null;
	s.drops.tick(nowMs());
	return s.drops.dps;
}

/** The latest frame for a (node, slot), or null when nothing is subscribed to it. */
export function latestFrame(node: string, slot: string): DataFrame | null {
	return slots.get(streamKey(node, slot))?.current ?? null;
}
