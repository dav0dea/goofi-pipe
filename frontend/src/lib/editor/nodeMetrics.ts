/**
 * Node geometry constants — these mirror the `--node-*` CSS custom properties in
 * `app.css`. They live here in JS too so the connector overlay can compute each
 * output port's y-position (which depends on how the output slots stack)
 * deterministically, without measuring the DOM.
 *
 * A node is drawn as two layers: a clipped visual `surface` (rounds the corners)
 * and an unclipped `ports` overlay (the connector pills, which overhang the
 * edges). The output handles can't be inside the clipped surface, so the overlay
 * positions them here using the same rhythm the SlotViewers lay out with:
 * header, then each slot — one unit tall collapsed, `unit + viewer` open.
 */
export const NODE = {
	width: 233,
	header: 36,
	unit: 24, // collapsed slot height === input connector pitch
	viewer: 144, // open viewer plot height
	border: 1
} as const;

/** An In/Out boundary pill's footprint (mirrors BoundaryNode.svelte's min-width +
 * padding/font). Pills are first-class draggables inside a sub-patch, so snapping
 * needs their real size — a node-sized fallback would be wildly off. */
export const BOUNDARY = {
	width: 96,
	height: 26
} as const;

/**
 * The rendered size of a node's surface box, computed from its slot layout — the
 * same rhythm GoofiNode draws with (header, then each output slot one unit tall
 * collapsed / `unit + viewer` open, with a floor of one input-connector unit per
 * input). This is the snap geometry's fallback when Svelte Flow hasn't measured a
 * node yet, and the accurate size for short sub-patch group nodes — for which the
 * old single fixed fallback was ~100px too tall, misaligning every snap.
 *
 * `outputExpanded[i]` is whether output slot i's inline viewer is open.
 */
export function nodeSurfaceSize(
	inputCount: number,
	outputExpanded: boolean[]
): { width: number; height: number } {
	const slotsStack = outputExpanded.reduce(
		(h, open) => h + (open ? NODE.unit + NODE.viewer : NODE.unit),
		0
	);
	const inputBody = Math.max(inputCount, 1) * NODE.unit;
	return { width: NODE.width, height: NODE.header + Math.max(slotsStack, inputBody) };
}
