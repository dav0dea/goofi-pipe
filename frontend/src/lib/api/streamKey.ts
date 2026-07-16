/** The stable routing key for a (node, slot) data stream. Shared by the
 * main-thread router (data.ts), the frame coalescer (frames.ts) and the worker
 * (dataWorker.ts) so the three can never drift out of agreement — a divergent
 * key would silently mis-route or duplicate a stream's subscribers. One stream
 * per (node, slot): every viewer kind shares it, so `kind` is NOT part of the key. */
export function streamKey(node: string, slot: string): string {
	return `${node} ${slot}`;
}
