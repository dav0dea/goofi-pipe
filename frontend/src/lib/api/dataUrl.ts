/** Build the data-plane WebSocket URL for a (node, slot) subscription. Mirrors the
 * bridge route `/data/<node>/<slot>` — ONE reduced stream per slot; the viewer kinds
 * a client draws are negotiated inband as ViewSpecs, not encoded in the URL. */
export function dataUrl(proto: string, host: string, node: string, slot: string): string {
	return `${proto}//${host}/data/${encodeURIComponent(node)}/${encodeURIComponent(slot)}`;
}
