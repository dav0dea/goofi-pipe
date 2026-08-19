/**
 * Data-plane transport (main-thread side of the Web Worker).
 *
 * The WS receive + GOOF decode + latest-wins coalescing all run in `dataWorker.ts`. This module is
 * the wire to it and nothing else: open a stream, close it, tell it what reduction to ask for, and
 * hand decoded frames to a single sink. The decoded frame's array buffer is transferred from the
 * worker (zero-copy).
 *
 * It deliberately holds NO idea of how many viewers want a stream. That belongs to the registry in
 * `frames.ts`, which is the one place that decides what the backend is asked for — see the
 * reconcile there for why a second opinion on the same question is what broke.
 */
import type { DataFrame } from '$lib/codec/decode';
import type { ViewSpec } from '$lib/viewers/capacity';

/** Where decoded frames go. One sink, registered once by `frames.ts`. */
type FrameSink = (node: string, slot: string, frame: DataFrame) => void;

let worker: Worker | null = null;
let sink: FrameSink | null = null;

/** Route decoded frames to `f`. Called once, at `frames.ts` module init. */
export function setFrameSink(f: FrameSink): void {
	sink = f;
}

function ensureWorker(): Worker {
	if (worker) return worker;
	worker = new Worker(new URL('./dataWorker.ts', import.meta.url), { type: 'module' });
	worker.addEventListener('message', (e: MessageEvent) => {
		const { node, slot, frame } = e.data as { node: string; slot: string; frame: DataFrame };
		sink?.(node, slot, frame);
	});
	return worker;
}

/** Open the `(node, slot)` stream. Idempotent at the worker: a stream already open stays open. */
export function openStream(node: string, slot: string): void {
	ensureWorker().postMessage({ op: 'sub', node, slot });
}

/** Close the `(node, slot)` stream and drop its socket. */
export function closeStream(node: string, slot: string): void {
	ensureWorker().postMessage({ op: 'unsub', node, slot });
}

/**
 * Ask the backend to reduce this stream to `specs` — every bound viewer's constraint, verbatim.
 * The bridge folds them (it needs the real frame to drop the ones a frame's shape rules out) and
 * across tabs, so the fold stays there; what this side owes it is a list that has actually changed.
 */
export function sendSpecs(node: string, slot: string, specs: ViewSpec[]): void {
	ensureWorker().postMessage({ op: 'spec', node, slot, specs });
}
