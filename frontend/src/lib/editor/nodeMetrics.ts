/** Node geometry, mirroring the `--node-*` custom properties in `app.css` so the connector
 * overlay can place ports without measuring the DOM. Keep the two in step. */
export const NODE = {
	width: 233,
	header: 36,
	unit: 24, // collapsed slot height === input connector pitch
	viewer: 144, // open viewer plot height
	border: 1
} as const;

/** A slot's height in units: a multi (list) slot is 2× tall. */
const slotUnits = (multi: boolean): number => (multi ? 2 : 1);

/** Total input-block height in units, floored at 1 so a node with no inputs still has a body. */
export function inputUnits(slots: string[], isMulti: (slot: string) => boolean): number {
	return Math.max(
		slots.reduce((n, s) => n + slotUnits(isMulti(s)), 0),
		1
	);
}

/** Vertical placement of each input connector; `top` is the centre (px) of the slot's block. */
export function inputPorts(
	slots: string[],
	isMulti: (slot: string) => boolean
): { slot: string; units: number; top: number }[] {
	let y = NODE.border + NODE.header;
	return slots.map((slot) => {
		const units = slotUnits(isMulti(slot));
		const top = y + (units * NODE.unit) / 2; // centre of the slot's block
		y += units * NODE.unit;
		return { slot, units, top };
	});
}

/** The rendered size of a node's surface box, the snap geometry's fallback until Svelte Flow
 * has measured the node. `outputExpanded[i]` is whether output slot i's inline viewer is open. */
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
