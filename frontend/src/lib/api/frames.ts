/**
 * The viewer registry, display-rate frame delivery, and a global paint scheduler (reports A16).
 *
 * The worker (`dataWorker.ts`) does the WS receive, GOOF decode, and latest-wins
 * coalescing. This main-thread layer adds ONE rAF-driven flush with a per-frame
 * time budget: when the worker delivers a frame it's stashed as the slot's
 * pending frame and a single rAF is requested; on that rAF we deliver pending
 * frames to viewers, most-starved-slot first, and once the budget (~8 ms,
 * leaving room for Svelte Flow + compositing) is spent we defer the rest to the
 * next frame. Latest-wins makes deferral free — a deferred slot just shows its
 * newest frame next tick — so no viewer is starved and no queue grows.
 *
 * It also records the latest frame per subscribed slot for the agent surface
 * (`latestFrame`). `subscribeFrames` / `latestFrame` keep their exact
 * signatures, so the viewer layer and agent surface are unchanged.
 */
import { closeStream, openStream, sendSpecs, setFrameSink } from './data';
import { paintDelay } from './paintCap';
import { perfStats } from './perfStats.svelte';
import { RateMeter } from './rateMeter';
import type { DataFrame } from '$lib/codec/decode';
import type { ViewSpec } from '$lib/viewers/capacity';
import { streamKey } from './streamKey';

type FrameCallback = (frame: DataFrame) => void;

/**
 * One viewer bound to a stream: where its frames go, and what it needs to draw.
 *
 * `spec` is null for a viewer that reads frames without constraining them — the metadata panel —
 * and for one that has not measured itself yet. A null contributes nothing to the reduction, which
 * is not the same as contributing an empty list: an empty LIST means full resolution.
 */
interface BoundViewer {
	cb: FrameCallback;
	spec: ViewSpec | null;
}

interface Slot {
	/** Every viewer bound to this stream, by its own stable token. THE registry: nothing else
	 *  counts viewers, and nothing else decides what the backend is asked for. */
	viewers: Map<string, BoundViewer>;
	/** Newest frame from the worker, not yet delivered to consumers. */
	pending: DataFrame | null;
	/** Latest frame delivered — what `latestFrame` returns. */
	current: DataFrame | null;
	/** When this slot last delivered, for most-starved-first fairness. */
	lastFlush: number;
	/** This stream's own coalescing rate. Per slot rather than app-wide: a drop belongs to the
	 * stream whose frame was overwritten, and summing them put a total beside an fps counter that
	 * is emphatically not a total — two numbers that could not be read together. */
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
	// The cap (paintCap.ts): inside the cooldown, hold the latest-wins frames on a TIMER — an
	// rAF here would fire at display rate just to decide "not yet", which is the exact battery
	// spend the cap exists to stop. The timer lands the cooldown; the rAF after it aligns the
	// actual paint to the next vsync.
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
	// Most-starved slot first (smallest lastFlush) so no slot is permanently
	// deferred by the budget; new slots (lastFlush 0) lead.
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
	// ONE paint per flush, not one per slot. This is the quantity the cap bounds (paintCap.ts caps
	// flush STARTS at 30/s) and the quantity the HUD names, so counting per slot made the readout
	// the sum of the open streams' rates — ~30 x N, a staircase climbing with every node added
	// while the cap itself never moved. A flush that painted nothing is not a paint.
	if (painted > 0) perfStats().delivered();
	if (dirty.size > 0) requestFlush(); // deferred slots → next frame
}


/**
 * What the backend has been told to serve for a stream: the encoded spec list. Absent means no
 * stream is open. This is the ONLY record of what the far end believes, and it is compared against
 * what the registry wants — so nothing is sent that would not change anything.
 */
const synced = new Map<string, string>();
const reconciling = new Set<string>();

/**
 * Bring the backend in line with the registry for one stream.
 *
 * Deferred to a microtask on purpose, and this is the whole design: every registry mutation asks
 * for a reconcile, and the reconcile reads the registry once the tick has SETTLED. A viewer that
 * detaches and re-attaches in the same tick — which a re-render does, and which one viewer of a
 * slot closing makes its siblings do — leaves the registry exactly as it found it, so the compare
 * below finds nothing to say and the socket is never touched. Sequencing the backend off each
 * individual mutation is what let a momentary count of zero tear down a stream other viewers were
 * still drawing from.
 */
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

/**
 * What this page needs a stream reduced to: every bound viewer's constraint, DISTINCT ones only.
 *
 * The bridge folds richest-per-dim, so a repeated constraint adds nothing to the result — and
 * dropping it here is what makes two equally-sized viewers of a slot a single demand. Without
 * that, closing either one rewrites the list and renegotiates a reduction that cannot change.
 */
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

/**
 * Bind a viewer to a (node, slot) stream. `token` identifies THIS viewer, so several viewers of
 * one slot collect rather than evict — a node's own viewer and a panel bound to the same slot are
 * two entries here and nothing else distinguishes them. Returns the unbind.
 *
 * `spec` is what this viewer needs the frame reduced to, or null for a reader that constrains
 * nothing. Re-binding with a changed spec is how a resize, a kind switch or a scroll out of view
 * is reported: one call, one registry, one reconcile.
 */
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
	// A viewer arriving on an open stream is invisible to the backend — the reconcile above finds
	// nothing to send — and the backend only speaks when something changed, so nothing is coming
	// to paint this one's first frame. The frame it needs is already here: replay it to the
	// joiner alone (the settled viewers have painted it; re-marking the slot dirty would repaint
	// them all). Without this, a panel joining a slot viewer's stream showed its empty state until
	// the producer's next emit — indefinitely for a stopped one.
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
	if (!s) return; // a frame for a stream nothing is bound to any more
	// A still-pending frame overwritten before it painted is a dropped frame (latest-wins) —
	// charged to THIS stream, which is the one that dropped it.
	if (s.pending !== null) s.drops.dropped();
	s.pending = frame;
	dirty.add(s);
	requestFlush();
});

/** Coalesced-frame rate for ONE stream — frames overwritten latest-wins before they painted,
 * measured per (node, slot). `null` for a slot with no live subscriber: absent is not zero, and a
 * panel reading `0/s` for a stream that is not running asserts something false. */
export function dropRate(node: string, slot: string): number | null {
	const s = slots.get(streamKey(node, slot));
	if (!s) return null;
	s.drops.tick(nowMs());
	return s.drops.dps;
}

/** The latest frame for a (node, slot) while it has at least one live subscriber.
 * Null for an unsubscribed (off-screen) slot. One stream per slot, so this is an
 * exact lookup. */
export function latestFrame(node: string, slot: string): DataFrame | null {
	return slots.get(streamKey(node, slot))?.current ?? null;
}
