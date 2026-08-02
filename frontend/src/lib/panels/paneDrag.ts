/**
 * The inspector pane's resize arithmetic — the part of the gesture that is not a DOM event.
 *
 * `InspectorOverlay.svelte` cannot be mounted in this repo's vitest, so whatever a unit test must
 * reach has to live in a `.ts` module. This is it.
 *
 * ONE shape for both anchors, which is the governing principle of the orientation work: the pane
 * sits against the right edge of its host or against the bottom one, with its grip on the leading
 * edge either way, so pushing the grip INTO the pane shrinks it on whichever axis it is on. Nothing
 * here is gated on a device or a pointer type, and the axis is not decided here — `@container
 * (orientation: portrait)` decides it and publishes the answer as `--pane-axis`, which the
 * component reads back rather than re-deriving.
 *
 * The FLOOR lives here. The CEILING deliberately does not: `max-width: min(30%, 30rem)` and
 * `max-height: 60%` are host- and rem-relative, so the stylesheet is the only place that can
 * evaluate them, and a number here would be a second answer to one question.
 */

export type PaneAxis = 'x' | 'y';

/**
 * Everything the AXIS selects, and the whole of it.
 *
 * A record rather than a condition, because the answer is looked up ONCE — at pointerdown — instead
 * of the same question ("is this the vertical one?") being asked again at every line of the gesture
 * that happens to need a dimension. Scattered that way it read as orientation threaded through the
 * drag; it is one fact, and this is where it is stated.
 *
 * ORIENTATION picks the record. INPUT MODALITY never touches it: `endsInDismiss` below is the whole
 * of what modality gates, and it takes no axis knowledge at all.
 */
export interface PaneAxisDims {
	/** The pane's floor on this axis, in px. The CEILING is deliberately elsewhere — see above. */
	min: number;
	/** Where a size dragged on this axis is remembered: one persistence idiom, one key per axis
	 *  (D-I3), so the two anchors cannot overwrite each other's. */
	key: string;
	/** The dimension of a box this axis sizes the pane by. */
	sizeOf(box: { width: number; height: number }): number;
}

export const PANE_AXES: Record<PaneAxis, PaneAxisDims> = {
	x: { min: 260, key: 'goofi.panelWidth', sizeOf: (b) => b.width },
	y: { min: 160, key: 'goofi.panelHeight', sizeOf: (b) => b.height }
};

export interface PaneDrag {
	/** The axis the grip drags, as the container query decided it. */
	axis: PaneAxis;
	/** The pane's RENDERED size on `axis` when the gesture began, in px — not its stored size, which
	 *  may sit above a ceiling only CSS can evaluate. */
	startSize: number;
	/** The pointer's coordinate on `axis` when the gesture began. */
	startPos: number;
	/** The floor for `axis`, in px. */
	min: number;
}

/** The coordinate this axis reads out of a pointer event, and only that one. */
export function coordOf(axis: PaneAxis, e: { clientX: number; clientY: number }): number {
	return axis === 'x' ? e.clientX : e.clientY;
}

/** The size a pointer at `at` is asking for, UNCLAMPED. */
function desiredSizeAt(drag: PaneDrag, at: number): number {
	return drag.startSize - (at - drag.startPos);
}

/** …and the size the pane is given for it. */
export function paneSizeAt(drag: PaneDrag, at: number): number {
	return Math.max(drag.min, desiredSizeAt(drag, at));
}

/**
 * How far PAST its floor a drag must pull the pane before the release closes it instead of resizing
 * it. A finger-width of deliberate overshoot: the pane clamps at the floor, so a resize that merely
 * bottoms out and a swipe meant to throw the pane away look identical on screen, and this is the
 * only thing that tells them apart.
 *
 * Its own number rather than `--hit`: that token is how big a tap TARGET must be, and it would move
 * for reasons that have nothing to do with how deliberate a swipe has to feel.
 */
export const DISMISS_OVERSHOOT_PX = 44;

/**
 * Does this gesture end in a dismiss rather than a resize? (D-I4.)
 *
 * The ONE thing input modality gates in the whole of this pane. The anchor, the clamp, the drag and
 * the persistence are all identical on a mouse and a finger; the swipe is layered on top of the same
 * gesture for touch alone, because a fine pointer already has the ✕ and always did. The ✕ stays in
 * both anchors regardless, so the pointer door never depends on a gesture.
 */
export function endsInDismiss(drag: PaneDrag, at: number, pointerType: string): boolean {
	return pointerType === 'touch' && desiredSizeAt(drag, at) <= drag.min - DISMISS_OVERSHOOT_PX;
}
