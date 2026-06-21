/**
 * Display-rate frame delivery on top of the worker data plane.
 *
 * The worker (`dataWorker.ts`) already does the WS receive, GOOF decode, and
 * latest-wins coalescing to ~display rate (reports A11/A13). This layer is a
 * thin main-thread router: it fans the worker's decoded frames out to the
 * viewer consumers and records the most recent frame per subscribed slot so the
 * agent / introspection surface can read "what is this viewer showing"
 * (`latestFrame`) without scraping the DOM. The cache holds a frame only while a
 * consumer is subscribed, so memory stays bounded to on-screen data.
 *
 * `subscribeFrames` / `latestFrame` keep their exact signatures, so the viewer
 * layer and agent surface are unchanged by the move to a worker.
 */
import { subscribeData } from './data';
import type { DataFrame } from '$lib/codec/decode';

type FrameCallback = (frame: DataFrame) => void;

interface Slot {
	/** Latest frame delivered to consumers — what `latestFrame` returns. */
	current: DataFrame | null;
	cbs: Set<FrameCallback>;
	unsub: () => void;
}

const slots = new Map<string, Slot>();

function key(node: string, slot: string): string {
	return `${node} ${slot}`;
}

/**
 * Subscribe to a (node, slot) stream, receiving the latest decoded frame at
 * ~display rate. Returns an unsubscribe function. Multiple consumers of the same
 * slot share one worker subscription.
 */
export function subscribeFrames(node: string, slot: string, cb: FrameCallback): () => void {
	const k = key(node, slot);
	let s = slots.get(k);
	if (!s) {
		const slot_: Slot = { current: null, cbs: new Set(), unsub: () => {} };
		slot_.unsub = subscribeData(node, slot, (frame) => {
			slot_.current = frame;
			for (const consumer of slot_.cbs) {
				try {
					consumer(frame);
				} catch (err) {
					console.error('frame consumer crashed', err);
				}
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
		cur.unsub();
		slots.delete(k);
	};
}

/** The latest frame for a slot, while it has at least one live subscriber.
 * Null for an unsubscribed (off-screen) slot. */
export function latestFrame(node: string, slot: string): DataFrame | null {
	return slots.get(key(node, slot))?.current ?? null;
}
