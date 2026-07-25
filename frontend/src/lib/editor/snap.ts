/** Shared snap-to-edges/center/gap logic used by both the node-drag flow
 * (NodeEditorPanel.svelte) and the new-node placement preview (PlacementPreview.svelte).
 *
 * The algorithm follows goofi3's `snapMultiDrag` shape: enumerate edge/center
 * pairings between the moving set and the static targets, plus a couple of
 * common gap offsets, pick the closest pair under SNAP_THRESHOLD, then collect
 * guides for any pairing within SNAP_RANGE so the visual hint fades in. */

export type Bounds = {
	left: number;
	right: number;
	top: number;
	bottom: number;
	cx: number;
	cy: number;
};

export type Guide = { x?: number; y?: number; opacity: number };

export const SNAP_THRESHOLD = 15; // engage snap within this many flow-units
export const SNAP_RANGE = 45; // start fading in guide hints from this far out
export const DEFAULT_NODE_W = 233; // fallback before a node has been measured
export const DEFAULT_NODE_H = 168; // a typical node: header + one open viewer (36 + 7u)

export function makeBounds(x: number, y: number, w: number, h: number): Bounds {
	return {
		left: x,
		right: x + w,
		top: y,
		bottom: y + h,
		cx: x + w / 2,
		cy: y + h / 2
	};
}

export function computeSnapDelta(
	draggedBounds: Bounds[],
	targets: Bounds[],
	altKey: boolean
): { dx: number; dy: number; guides: Guide[] } {
	if (altKey || draggedBounds.length === 0 || targets.length === 0) {
		return { dx: 0, dy: 0, guides: [] };
	}

	const V_GAPS = [0, DEFAULT_NODE_H * 0.5];
	const H_GAPS = [0, DEFAULT_NODE_W * 0.25];
	let bestDistY = Infinity;
	let bestDy = 0;
	let bestDistX = Infinity;
	let bestDx = 0;

	for (const me of draggedBounds) {
		for (const oe of targets) {
			for (const gap of V_GAPS) {
				const yPairs: [number, number][] = [
					[me.top, oe.top],
					[me.bottom, oe.bottom],
					[me.top, oe.bottom + gap],
					[me.bottom, oe.top - gap]
				];
				if (gap === 0) yPairs.push([me.cy, oe.cy]);
				for (const [myE, otherE] of yPairs) {
					const d = Math.abs(myE - otherE);
					if (d < SNAP_THRESHOLD && d < bestDistY) {
						bestDistY = d;
						bestDy = otherE - myE;
					}
				}
			}
			for (const gap of H_GAPS) {
				const xPairs: [number, number][] = [
					[me.left, oe.left],
					[me.right, oe.right],
					[me.left, oe.right + gap],
					[me.right, oe.left - gap]
				];
				if (gap === 0) xPairs.push([me.cx, oe.cx]);
				for (const [myE, otherE] of xPairs) {
					const d = Math.abs(myE - otherE);
					if (d < SNAP_THRESHOLD && d < bestDistX) {
						bestDistX = d;
						bestDx = otherE - myE;
					}
				}
			}
		}
	}

	const dx = bestDistX < Infinity ? bestDx : 0;
	const dy = bestDistY < Infinity ? bestDy : 0;

	const guides: Guide[] = [];
	for (const me of draggedBounds) {
		const shifted: Bounds = {
			left: me.left + dx,
			right: me.right + dx,
			top: me.top + dy,
			bottom: me.bottom + dy,
			cx: me.cx + dx,
			cy: me.cy + dy
		};
		for (const oe of targets) {
			for (const gap of V_GAPS) {
				const yPairs: [number, number][] = [
					[shifted.top, oe.top],
					[shifted.bottom, oe.bottom],
					[shifted.top, oe.bottom + gap],
					[shifted.bottom, oe.top - gap]
				];
				if (gap === 0) yPairs.push([shifted.cy, oe.cy]);
				for (const [myE, otherE] of yPairs) {
					const d = Math.abs(myE - otherE);
					if (d < SNAP_RANGE) {
						guides.push({ y: otherE, opacity: d < 0.5 ? 1 : 1 - d / SNAP_RANGE });
					}
				}
			}
			for (const gap of H_GAPS) {
				const xPairs: [number, number][] = [
					[shifted.left, oe.left],
					[shifted.right, oe.right],
					[shifted.left, oe.right + gap],
					[shifted.right, oe.left - gap]
				];
				if (gap === 0) xPairs.push([shifted.cx, oe.cx]);
				for (const [myE, otherE] of xPairs) {
					const d = Math.abs(myE - otherE);
					if (d < SNAP_RANGE) {
						guides.push({ x: otherE, opacity: d < 0.5 ? 1 : 1 - d / SNAP_RANGE });
					}
				}
			}
		}
	}

	return { dx, dy, guides };
}
