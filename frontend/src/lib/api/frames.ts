/**
 * Display-rate frame delivery on top of the raw data plane.
 *
 * `api/data.ts` fans out every decoded frame the WS delivers. Viewers don't need
 * every frame — they need the *latest* one, at most once per paint. This layer
 * coalesces delivery to one frame per animation frame per (node, slot) and drops
 * the intermediate frames, matching the iceoryx2 latest-wins semantics and
 * keeping the main thread free under bursty kHz / HD-video loads.
 *
 * It also records the most recent frame per subscribed slot, so the agent /
 * introspection surface can read "what is this viewer currently showing"
 * (`latestFrame`) without scraping the DOM. The cache holds a frame only while
 * at least one consumer is subscribed, so memory stays bounded to on-screen data.
 */
import { subscribeData } from './data';
import type { DataFrame } from '$lib/codec/decode';

type FrameCallback = (frame: DataFrame) => void;

interface Slot {
	/** Most recent frame delivered by the WS, not yet flushed to consumers. */
	pending: DataFrame | null;
	/** Latest frame flushed to consumers — what `latestFrame` returns. */
	current: DataFrame | null;
	cbs: Set<FrameCallback>;
	raf: number | null;
	unsubRaw: () => void;
}

const slots = new Map<string, Slot>();

function key(node: string, slot: string): string {
	return `${node} ${slot}`;
}

const scheduleFlush =
	typeof requestAnimationFrame === 'function'
		? (fn: () => void): number => requestAnimationFrame(fn)
		: (fn: () => void): number => setTimeout(fn, 16) as unknown as number;

const cancelFlush =
	typeof cancelAnimationFrame === 'function'
		? (id: number): void => cancelAnimationFrame(id)
		: (id: number): void => clearTimeout(id);

/**
 * Subscribe to a (node, slot) stream, receiving at most one (latest) frame per
 * animation frame. Returns an unsubscribe function. Multiple consumers of the
 * same slot share one underlying WS and one coalescing timer.
 */
export function subscribeFrames(node: string, slot: string, cb: FrameCallback): () => void {
	const k = key(node, slot);
	let s = slots.get(k);
	if (!s) {
		const slot_: Slot = {
			pending: null,
			current: null,
			cbs: new Set(),
			raf: null,
			unsubRaw: () => {}
		};
		slot_.unsubRaw = subscribeData(node, slot, (f) => {
			slot_.pending = f;
			if (slot_.raf === null) {
				slot_.raf = scheduleFlush(() => {
					slot_.raf = null;
					const frame = slot_.pending;
					slot_.pending = null;
					if (!frame) return;
					slot_.current = frame;
					for (const consumer of slot_.cbs) {
						try {
							consumer(frame);
						} catch (err) {
							console.error('frame consumer crashed', err);
						}
					}
				});
			}
		});
		s = slot_;
		slots.set(k, s);
	}
	s.cbs.add(cb);
	return () => {
		const cur = slots.get(k);
		if (!cur) return;
		cur.cbs.delete(cb);
		if (cur.cbs.size > 0) return;
		// Last consumer gone: tear down the WS, the timer, and the cached frame.
		if (cur.raf !== null) cancelFlush(cur.raf);
		cur.unsubRaw();
		slots.delete(k);
	};
}

/** The latest frame for a slot, while it has at least one live subscriber.
 * Null for an unsubscribed (off-screen) slot. */
export function latestFrame(node: string, slot: string): DataFrame | null {
	return slots.get(key(node, slot))?.current ?? null;
}
