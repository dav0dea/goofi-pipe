/**
 * Data-plane host (main-thread side of the Web Worker).
 *
 * The WS receive + GOOF decode + latest-wins coalescing all run in
 * `dataWorker.ts` (reports A11/A13). This module owns the single worker, routes
 * its decoded frames to the per-(node, slot) consumers, and refcounts
 * subscriptions so the worker opens/closes the underlying WS. The decoded
 * frame's array buffer is transferred from the worker (zero-copy).
 */
import type { DataFrame } from '$lib/codec/decode';

type FrameCallback = (frame: DataFrame) => void;

let worker: Worker | null = null;
const consumers = new Map<string, Set<FrameCallback>>();

function key(node: string, slot: string, kind: string): string {
	return `${node} ${slot} ${kind}`;
}

function ensureWorker(): Worker {
	if (worker) return worker;
	worker = new Worker(new URL('./dataWorker.ts', import.meta.url), { type: 'module' });
	worker.addEventListener('message', (e: MessageEvent) => {
		const { node, slot, kind, frame } = e.data as {
			node: string;
			slot: string;
			kind: string;
			frame: DataFrame;
		};
		const set = consumers.get(key(node, slot, kind));
		if (!set) return;
		for (const cb of set) {
			try {
				cb(frame);
			} catch (err) {
				console.error('data frame consumer crashed', err);
			}
		}
	});
	return worker;
}

/**
 * Subscribe to a (node, slot) data stream rendered for viewer `kind`. Returns an
 * unsubscribe function. Frames arrive already decoded (and adapted to `kind` on the
 * bridge) and coalesced to ~display rate by the worker. Multiple consumers of the
 * same (node, slot, kind) share one worker WS; different kinds get their own.
 */
export function subscribeData(node: string, slot: string, kind: string, cb: FrameCallback): () => void {
	const k = key(node, slot, kind);
	let set = consumers.get(k);
	if (!set) {
		set = new Set();
		consumers.set(k, set);
		ensureWorker().postMessage({ op: 'sub', node, slot, kind });
	}
	set.add(cb);
	return () => {
		const s = consumers.get(k);
		if (!s) return;
		s.delete(cb);
		if (s.size === 0) {
			consumers.delete(k);
			worker?.postMessage({ op: 'unsub', node, slot, kind });
		}
	};
}
