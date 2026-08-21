/** The stable routing key for a (node, slot) data stream. One stream per slot, so the viewer
 * `kind` is NOT part of it. */
export function streamKey(node: string, slot: string): string {
	return `${node} ${slot}`;
}
