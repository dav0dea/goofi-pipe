/** Build the data-plane WebSocket URL for a (node, slot) subscription. */
export function dataUrl(proto: string, host: string, node: string, slot: string): string {
	return `${proto}//${host}/data/${encodeURIComponent(node)}/${encodeURIComponent(slot)}`;
}
