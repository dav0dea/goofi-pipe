import { inputPorts } from './nodeMetrics';

/** The proximity reveal for input-slot names, as arithmetic in flow space — no DOM measurement on
 * a pointermove path, and no invalidation when the canvas pans or zooms mid-drag. */

/** One input connector's flow-space centre, under the key the editor publishes it as. */
export interface SlotAnchor {
	key: string;
	x: number;
	y: number;
}

export interface AnchorNode {
	uid: string;
	/** Flow-space position of the node's top-left corner. */
	x: number;
	y: number;
	/** Input slot ids, in declaration order. */
	slots: string[];
	/** Which of them are MULTI (list) slots. */
	multi: ReadonlySet<string>;
}

/** How close the pointer must come, in SCREEN px, for an input to name itself: one coarse `--hit`.
 * Screen px because a fingertip is physical — the caller divides by the live zoom. */
export const SLOT_PROXIMITY_PX = 44;

/** Every input connector of every node, keyed by the caller's own key format. */
export function inputAnchors(
	nodes: readonly AnchorNode[],
	key: (uid: string, slot: string) => string
): SlotAnchor[] {
	const out: SlotAnchor[] = [];
	for (const n of nodes) {
		for (const p of inputPorts(n.slots, (slot) => n.multi.has(slot)))
			out.push({ key: key(n.uid, p.slot), x: n.x, y: n.y + p.top });
	}
	return out;
}

/** The anchors within `radius` of `p`, measured in the anchors' own space, inclusive at the edge. */
export function nearSlots(
	anchors: readonly SlotAnchor[],
	p: { x: number; y: number },
	radius: number
): Set<string> {
	const r2 = radius * radius;
	const out = new Set<string>();
	for (const a of anchors) {
		const dx = a.x - p.x;
		const dy = a.y - p.y;
		if (dx * dx + dy * dy <= r2) out.add(a.key);
	}
	return out;
}

/** Whether two key sets hold the same members, so a fresh Set per pointermove invalidates nothing. */
export function sameKeys(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
	if (a.size !== b.size) return false;
	for (const k of a) if (!b.has(k)) return false;
	return true;
}
