/** The stable routing key for a (node, slot, kind) data stream. Shared by the
 * main-thread router (data.ts), the frame coalescer (frames.ts) and the worker
 * (dataWorker.ts) so the three can never drift out of agreement — a divergent
 * key would silently mis-route or duplicate a stream's subscribers. */
export function streamKey(node: string, slot: string, kind: string): string {
	return `${node} ${slot} ${kind}`;
}
