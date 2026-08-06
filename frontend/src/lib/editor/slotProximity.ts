import { inputPorts } from './nodeMetrics';

/**
 * The proximity reveal for input-slot names, as arithmetic.
 *
 * `.conn-label` on a connector pill is the ONLY rendering of an input slot's name. A mouse asks for
 * it by hovering the pill; a finger cannot, so R rested every one of them open under a coarse
 * pointer — which put a name tag on every input on the canvas, permanently. This replaces that with
 * a door BOTH modalities get: while a cable is in flight, the inputs the pointer is closing on name
 * themselves, and nothing names itself the rest of the time.
 *
 * Deliberately DOM-free. `NodeEditorPanel` already holds every node's flow-space position, and
 * `nodeMetrics` already computes each connector's offset within it — the same numbers the cable
 * anchors are drawn from — so the whole reveal is arithmetic over state the editor has, with not one
 * `getBoundingClientRect` on a pointermove path. That is also what makes it pan/zoom-proof: the
 * anchors are snapshotted once in FLOW space, and each move converts the pointer into that space
 * through the live viewport, so a canvas that pans or zooms mid-drag needs no invalidation.
 */

/** One input connector's flow-space centre, under the key the editor publishes it as. */
export interface SlotAnchor {
	key: string;
	x: number;
	y: number;
}

/** What {@link inputAnchors} needs of a node: where it sits, and what it accepts. */
export interface AnchorNode {
	uid: string;
	/** Flow-space position of the node's top-left corner. */
	x: number;
	y: number;
	/** Input slot ids, in declaration order (the order they stack down the left edge). */
	slots: string[];
	/** Which of them are MULTI (list) slots — those are two units tall. */
	multi: ReadonlySet<string>;
}

/**
 * How close the pointer must come, in SCREEN px, for an input to name itself.
 *
 * One coarse `--hit` (app.css raises `--hit` to 44px under the coarse idiom). The radius answers
 * "which inputs is this drop choosing between", and a fingertip target IS that neighbourhood — so
 * the tags that light up are exactly the ones a finger could plausibly land on. Connectors tile the
 * left edge at `--node-u` (24px), so 44px reaches about two slots either side: enough to tell them
 * apart, far short of naming a whole node. Screen px rather than flow px because a fingertip is a
 * physical size — the caller divides by the live zoom before comparing against flow-space anchors.
 */
export const SLOT_PROXIMITY_PX = 44;

/** Every input connector of every node, keyed by the caller's own key format. */
export function inputAnchors(
	nodes: readonly AnchorNode[],
	key: (uid: string, slot: string) => string
): SlotAnchor[] {
	const out: SlotAnchor[] = [];
	for (const n of nodes) {
		// `.conn.in` hugs the node's left edge, centred on its slot block — the same `top` the cable
		// anchors are measured from, so the reveal and the cable it reveals for agree by construction.
		for (const p of inputPorts(n.slots, (slot) => n.multi.has(slot)))
			out.push({ key: key(n.uid, p.slot), x: n.x, y: n.y + p.top });
	}
	return out;
}

/** The anchors within `radius` of `p`, measured in the anchors' own space. Inclusive at the
 *  boundary: the radius names how far the reveal reaches, and a strict `<` would leave a point
 *  exactly on it to floating-point noise. */
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

/** Whether two key sets hold the same members. The publisher mints a fresh Set on every
 *  pointermove; without this every move would invalidate every node on the canvas for a set that
 *  usually did not change. */
export function sameKeys(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
	if (a.size !== b.size) return false;
	for (const k of a) if (!b.has(k)) return false;
	return true;
}
