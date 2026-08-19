/**
 * Display-rate frame delivery + a global paint scheduler (reports A16).
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
import { subscribeData } from './data';
import { paintDelay } from './paintCap';
import { perfStats } from './perfStats.svelte';
import { RateMeter } from './rateMeter';
import type { DataFrame } from '$lib/codec/decode';
import { streamKey } from './streamKey';

type FrameCallback = (frame: DataFrame) => void;

interface Slot {
	/** Newest frame from the worker, not yet delivered to consumers. */
	pending: DataFrame | null;
	/** Latest frame delivered — what `latestFrame` returns. */
	current: DataFrame | null;
	cbs: Set<FrameCallback>;
	unsub: () => void;
	/** When this slot last delivered, for most-starved-first fairness. */
	lastFlush: number;
	/** This stream's own coalescing rate. Per slot rather than app-wide: a drop belongs to the
	 * stream whose frame was overwritten, and summing them put a total beside an fps counter that
	 * is emphatically not a total — two numbers that could not be read together. */
	drops: RateMeter;
	/** Pending teardown, armed when the last consumer left and cancelled if one returns. */
	linger: ReturnType<typeof setTimeout> | null;
}

const slots = new Map<string, Slot>();
const dirty = new Set<Slot>();
const FRAME_BUDGET_MS = 8;
/**
 * How long a stream outlives its last consumer before it is torn down.
 *
 * A viewer detaching is NOT evidence that the slot is unwatched: an effect re-run detaches and
 * re-attaches the same viewer, and a re-render of a viewer's host does the same — both within a
 * tick. Tearing down on that transient zero closed the WS and dropped the cached frame, and every
 * OTHER viewer of the slot then went to its empty state: collapsing a node's own viewer blanked
 * the viewer PANEL bound to the same slot. Long enough to span a re-render across a frame or two,
 * short enough that a genuinely closed viewer's socket does not linger perceptibly.
 */
const LINGER_MS = 250;

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
		for (const consumer of s.cbs) {
			try {
				consumer(frame);
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
 * Subscribe to a (node, slot) stream, receiving the latest decoded frame at
 * ~display rate (subject to the global paint budget). Returns an unsubscribe
 * function. Multiple consumers of the same (node, slot) — of any viewer kind —
 * share one worker subscription (ONE reduced stream per slot).
 */
export function subscribeFrames(node: string, slot: string, cb: FrameCallback): () => void {
	const k = streamKey(node, slot);
	let s = slots.get(k);
	if (!s) {
		const slot_: Slot = {
			pending: null,
			current: null,
			cbs: new Set(),
			unsub: () => {},
			lastFlush: 0,
			drops: new RateMeter(nowMs()),
			linger: null
		};
		slot_.unsub = subscribeData(node, slot, (frame) => {
			// A still-pending frame overwritten before it painted is a dropped frame
			// (latest-wins) — charged to THIS stream, which is the one that dropped it.
			if (slot_.pending !== null) slot_.drops.dropped();
			slot_.pending = frame; // overwrite — latest wins
			dirty.add(slot_);
			requestFlush();
		});
		s = slot_;
		slots.set(k, s);
	}
	// A consumer returning inside the linger window claims the stream back, cache and socket
	// intact — which is what makes the detach/re-attach of a re-render cost nothing.
	if (s.linger !== null) {
		clearTimeout(s.linger);
		s.linger = null;
	}
	s.cbs.add(cb);
	// A joiner of an already-open slot is invisible to the bridge — no worker traffic happens —
	// and the bridge only sends when something changed, so nothing is coming to paint this
	// consumer's first frame. The frame it needs is already cached here: replay it to the
	// joiner alone (the settled consumers have painted it; re-marking the slot dirty would
	// repaint them all). Without this, a metadata panel joining a slot viewer's stream showed
	// its empty state until the producer's next emit — indefinitely for a stopped one.
	if (s.current) {
		try {
			cb(s.current);
		} catch (err) {
			console.error('frame consumer crashed', err);
		}
	}
	return () => {
		const cur = slots.get(k);
		if (!cur) return;
		cur.cbs.delete(cb);
		if (cur.cbs.size > 0 || cur.linger !== null) return;
		cur.linger = setTimeout(() => {
			// Re-read: the slot may have been reclaimed and torn down again while this waited.
			if (slots.get(k) !== cur || cur.cbs.size > 0) return;
			cur.unsub();
			dirty.delete(cur);
			slots.delete(k);
		}, LINGER_MS);
	};
}

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
