/** Placing a new node with a finger — the gesture, with no DOM in it. Modality is asked of the
 * EVENT, never a media query: a hybrid laptop carries a trackpad and a touchscreen at once. */

export interface PlacementPointer {
	pointerId: number;
	pointerType: string;
	clientX: number;
	clientY: number;
}

export interface PlacementPoint {
	x: number;
	y: number;
}

export interface TouchPlacement {
	/** True while a gesture this recognizer took is in flight. */
	readonly active: boolean;
	/** Take this pointerdown, or leave it. Returns where the ghost goes; null if it is not ours. */
	down(e: PlacementPointer, inCanvas: boolean): PlacementPoint | null;
	/** Where the ghost goes now; null if this move belongs to some other pointer. */
	move(e: PlacementPointer): PlacementPoint | null;
	/** Where to commit; null if this up belongs to some other pointer. Ends the gesture. */
	up(e: PlacementPointer): PlacementPoint | null;
	/** End the gesture without committing — the placement stays pending, the ghost stays put. */
	cancel(e: PlacementPointer): void;
}

/** How the ghost hangs off the point carrying it. */
export type GhostAnchor = 'top-left' | 'centre';

/** Where the ghost's TOP-LEFT goes, for a ghost of `size` carried by the point `at`. Both arguments
 * are in FLOW units, so the offset applies once to the position and never to a CSS transform. */
export function ghostOrigin(
	at: PlacementPoint,
	size: { w: number; h: number },
	anchor: GhostAnchor
): PlacementPoint {
	if (anchor === 'top-left') return { x: at.x, y: at.y };
	return { x: at.x - size.w / 2, y: at.y - size.h / 2 };
}

export function createTouchPlacement(): TouchPlacement {
	// Held by id, so a SECOND finger landing mid-drag cannot teleport the ghost.
	let held: number | null = null;
	const mine = (e: PlacementPointer): boolean => held !== null && e.pointerId === held;
	const point = (e: PlacementPointer): PlacementPoint => ({ x: e.clientX, y: e.clientY });

	return {
		get active(): boolean {
			return held !== null;
		},
		down(e, inCanvas) {
			if (e.pointerType !== 'touch' || !inCanvas || held !== null) return null;
			held = e.pointerId;
			return point(e);
		},
		move(e) {
			return mine(e) ? point(e) : null;
		},
		up(e) {
			if (!mine(e)) return null;
			held = null;
			return point(e);
		},
		cancel(e) {
			if (mine(e)) held = null;
		}
	};
}
