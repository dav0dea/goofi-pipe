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
		const slot_: Slot = { pending: null, current: null, cbs: new Set(), unsub: () => {}, lastFlush: 0 };
		slot_.unsub = subscribeData(node, slot, (frame) => {
			// A still-pending frame overwritten before it painted is a dropped frame
			// (latest-wins) — surface that to the perf HUD.
			if (slot_.pending !== null) perfStats().dropped();
			slot_.pending = frame; // overwrite — latest wins
			dirty.add(slot_);
			requestFlush();
		});
		s = slot_;
		slots.set(k, s);
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
		if (cur.cbs.size > 0) return;
		cur.unsub();
		dirty.delete(cur);
		slots.delete(k);
	};
}

/** The latest frame for a (node, slot) while it has at least one live subscriber.
 * Null for an unsubscribed (off-screen) slot. One stream per slot, so this is an
 * exact lookup. */
export function latestFrame(node: string, slot: string): DataFrame | null {
	return slots.get(streamKey(node, slot))?.current ?? null;
}
