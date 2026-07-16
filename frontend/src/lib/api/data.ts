/**
 * Data-plane host (main-thread side of the Web Worker).
 *
 * The WS receive + GOOF decode + latest-wins coalescing all run in
 * `dataWorker.ts`. This module owns the single worker, routes its decoded frames
 * to the per-(node, slot) consumers, and refcounts subscriptions so the worker
 * opens/closes the underlying WS. ONE stream per (node, slot) serves every viewer
 * kind: viewers contribute ViewSpecs which are sent as a LIST inband; the bridge
 * merges them against the real frame and reduces the slot once. The decoded
 * frame's array buffer is transferred from the worker (zero-copy).
 */
import type { DataFrame } from '$lib/codec/decode';
import type { ViewSpec } from '$lib/viewers/capacity';
import { streamKey } from './streamKey';

type FrameCallback = (frame: DataFrame) => void;

let worker: Worker | null = null;
const consumers = new Map<string, Set<FrameCallback>>();
// Per (node, slot): each viewer's contributed ViewSpec, keyed by a stable consumer
// token. Multiple viewers of one slot in a tab share ONE WS, so we collect their
// specs here and send the union as a list to the bridge (which folds — it needs the
// real frame to drop incompatible viewers — and also folds across tabs).
const specs = new Map<string, Map<string, ViewSpec>>();


function ensureWorker(): Worker {
	if (worker) return worker;
	worker = new Worker(new URL('./dataWorker.ts', import.meta.url), { type: 'module' });
	worker.addEventListener('message', (e: MessageEvent) => {
		const { node, slot, frame } = e.data as {
			node: string;
			slot: string;
			frame: DataFrame;
		};
		const set = consumers.get(streamKey(node, slot));
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
 * Subscribe to a (node, slot) data stream. Returns an unsubscribe function. Frames
 * arrive already decoded and coalesced to ~display rate by the worker, reduced to
 * the union of the slot's viewers' ViewSpecs. Multiple consumers of the same
 * (node, slot) — of any kind — share one worker WS.
 */
export function subscribeData(node: string, slot: string, cb: FrameCallback): () => void {
	const k = streamKey(node, slot);
	let set = consumers.get(k);
	if (!set) {
		set = new Set();
		consumers.set(k, set);
		ensureWorker().postMessage({ op: 'sub', node, slot });
	}
	set.add(cb);
	return () => {
		const s = consumers.get(k);
		if (!s) return;
		s.delete(cb);
		if (s.size === 0) {
			consumers.delete(k);
			worker?.postMessage({ op: 'unsub', node, slot });
		}
	};
}

// Last posted specs per key, and keys with a flush already queued. Coalescing the
// post to a microtask collapses the clear→re-add pair that a ViewerFeed resize
// fires in one tick (cleanup runs clearViewSpec, body runs setViewSpec) into ONE
// post of the final list — no transient empty list reaching the bridge (which would
// ship one full-resolution frame), and no redundant re-post of an unchanged list.
const lastSpecs = new Map<string, string>();
const flushQueued = new Set<string>();

function pushSpecs(node: string, slot: string, k: string): void {
	if (flushQueued.has(k)) return;
	flushQueued.add(k);
	queueMicrotask(() => {
		flushQueued.delete(k);
		const m = specs.get(k);
		const list = m ? [...m.values()] : [];
		const sig = JSON.stringify(list);
		if (lastSpecs.get(k) === sig) return; // unchanged → nothing to renegotiate
		if (list.length === 0) lastSpecs.delete(k); // slot drained — don't grow the map
		else lastSpecs.set(k, sig);
		ensureWorker().postMessage({ op: 'spec', node, slot, specs: list });
	});
}

/**
 * Contribute (or update) this viewer's ViewSpec for a (node, slot) stream. `token`
 * identifies the viewer so several viewers of one slot collect (not overwrite). The
 * list is sent inband over the data WS; the bridge merges + reduces each frame to
 * it. Call after `subscribeData` (ViewerFeed always subscribes in a source-earlier
 * effect); the worker re-sends the specs on every (re)connect.
 */
export function setViewSpec(node: string, slot: string, token: string, spec: ViewSpec): void {
	const k = streamKey(node, slot);
	let m = specs.get(k);
	if (!m) {
		m = new Map();
		specs.set(k, m);
	}
	m.set(token, spec);
	pushSpecs(node, slot, k);
}

/** Drop this viewer's ViewSpec contribution (on unsubscribe / kind change). */
export function clearViewSpec(node: string, slot: string, token: string): void {
	const k = streamKey(node, slot);
	const m = specs.get(k);
	if (!m) return;
	m.delete(token);
	if (m.size === 0) specs.delete(k);
	// Re-post even when empty: the WS refcount (subscribeData) is decoupled from the
	// spec refcount, so the slot may outlive this contribution. pushSpecs with no
	// specs posts the empty list, clearing the worker's stale specs instead of
	// leaving them to reduce a later viewer's frames to this one's pixel budget.
	pushSpecs(node, slot, k);
}
