/** Build the data-plane WebSocket URL for a (node, slot, kind) subscription.
 * Mirrors the bridge route `/data/<node>/<slot>/<kind>` (viewer adapters). */
export function dataUrl(
	proto: string,
	host: string,
	node: string,
	slot: string,
	kind: string
): string {
	return `${proto}//${host}/data/${encodeURIComponent(node)}/${encodeURIComponent(slot)}/${encodeURIComponent(kind)}`;
}
