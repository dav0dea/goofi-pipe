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

/** Total input-block height in units — a single slot is 1, a MULTI (list) slot 2,
 * floored at 1 so a node with no inputs still has a body. The input-side height
 * GoofiNode reserves and the snap-geometry fallback uses. */
export function inputUnits(slots: string[], isMulti: (slot: string) => boolean): number {
	return Math.max(
		slots.reduce((n, s) => n + (isMulti(s) ? 2 : 1), 0),
		1
	);
}

/** Vertical placement of each input connector. Slots stack from below the header in
 * declaration order; a MULTI (list) slot is 2 units tall, a single slot 1. `top` is
 * the centre (px) of each slot's block — the rhythm GoofiNode's overlay draws with. */
export function inputPorts(
	slots: string[],
	isMulti: (slot: string) => boolean
): { slot: string; units: number; top: number }[] {
	let y = NODE.border + NODE.header;
	return slots.map((slot) => {
		const units = isMulti(slot) ? 2 : 1;
		const top = y + (units * NODE.unit) / 2; // centre of the slot's block
		y += units * NODE.unit;
		return { slot, units, top };
	});
}

/**
 * The rendered size of a node's surface box, computed from its slot layout — the
 * same rhythm GoofiNode draws with (header, then each output slot one unit tall
 * collapsed / `unit + viewer` open, with a floor of one unit for the input body).
 * This is the snap geometry's fallback when Svelte Flow hasn't measured a node yet,
 * and the accurate size for short sub-patch group nodes — for which the old single
 * fixed fallback was ~100px too tall, misaligning every snap.
 *
 * `inputUnitsTotal` is the input body's height in units (a multi slot counts as 2 —
 * see {@link inputUnits}); `outputExpanded[i]` is whether output slot i's inline
 * viewer is open.
 */
export function nodeSurfaceSize(
	inputUnitsTotal: number,
	outputExpanded: boolean[]
): { width: number; height: number } {
	const slotsStack = outputExpanded.reduce(
		(h, open) => h + (open ? NODE.unit + NODE.viewer : NODE.unit),
		0
	);
	const inputBody = Math.max(inputUnitsTotal, 1) * NODE.unit;
	return { width: NODE.width, height: NODE.header + Math.max(slotsStack, inputBody) };
}
