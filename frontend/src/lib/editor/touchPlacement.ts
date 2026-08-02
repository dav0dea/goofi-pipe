/**
 * Placing a new node WITH A FINGER — the gesture, with no DOM in it.
 *
 * `PlacementPreview.svelte` cannot be mounted in this repo's vitest, so whatever a unit test must
 * reach lives in a `.ts` module; `editor/longPress.ts` and `editor/eventPoint.ts` set the idiom.
 *
 * ORIENTATION decides an anchored AXIS; INPUT MODALITY decides the GESTURE — the rule
 * `panels/paneDrag.ts` states and `tests/e2e/tests/touch-placement.spec.ts` re-measures in both
 * anchors. So modality is asked of the EVENT (`pointerType === 'touch'`) and never of a media
 * query: a hybrid laptop carries a trackpad and a screen at once, and each has to stay right while
 * the other is plugged in. A media query answers for the DEVICE and would get one of them wrong.
 *
 * THERE IS NO TAP-VS-DRAG CLASSIFICATION, deliberately. A tap is a drag of zero length: `down` puts
 * the ghost under the finger and `up` commits wherever it ended, so both are ONE path and there is
 * no threshold anything can fall on the wrong side of.
 *
 * Only a pointerdown INSIDE the canvas is taken, and that is what keeps the cancel path single: a
 * touch outside is left entirely alone, so the synthetic click it produces reaches the same window
 * handler a mouse click does and cancels the placement there. `active` is the one flag that says
 * "this gesture is mine" — the preview's `touchstart` blocker reads it too, so what it holds back
 * from SvelteFlow's panner is exactly the gesture this recognizer is already driving.
 */

/** The only fields this reads off a `PointerEvent`; supplied structurally by a real one. */
export interface PlacementPointer {
	pointerId: number;
	pointerType: string;
	clientX: number;
	clientY: number;
}

/** A client-space point, in the shape the preview stores the ghost's position in. */
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

export function createTouchPlacement(): TouchPlacement {
	// The finger that took the gesture. Held by id so a SECOND finger landing mid-drag is ignored
	// rather than allowed to teleport the ghost across the canvas.
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
