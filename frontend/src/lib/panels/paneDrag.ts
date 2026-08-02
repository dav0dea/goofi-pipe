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
 * NEITHER BOUND lives here, and that is deliberate. The ceilings are host- and rem-relative, so
 * only the stylesheet can evaluate them; the floor is declared beside them and written into them,
 * so the pane's allowed range cannot come out empty. `paneMin` below is where the published floor
 * becomes the number this arithmetic clamps to — and why it is published at all.
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
	/** Where a size dragged on this axis is remembered: one persistence idiom, one key per axis
	 *  (D-I3), so the two anchors cannot overwrite each other's. */
	key: string;
	/** The dimension of a box this axis sizes the pane by. */
	sizeOf(box: { width: number; height: number }): number;
}

export const PANE_AXES: Record<PaneAxis, PaneAxisDims> = {
	x: { key: 'goofi.panelWidth', sizeOf: (b) => b.width },
	y: { key: 'goofi.panelHeight', sizeOf: (b) => b.height }
};

/**
 * The pane's FLOOR on whichever axis it is anchored on, out of the `--pane-min` it publishes.
 *
 * The floor used to be a number in `PANE_AXES` above, one file away from the ceiling that bounds
 * it — and the two crossed, which cost the landscape pane its entire resize range.
 * `InspectorOverlay.svelte`'s `--pane-min` declaration is where that is written down and fixed; the
 * consequence here is that the floor is READ, exactly as the anchor is, rather than restated.
 */
export function paneMin(declared: string): number {
	const px = parseFloat(declared);
	// A pane that publishes no floor is a broken stylesheet, not a runtime condition — and every
	// clamp above would silently become NaN, which reads on screen as a drag that does nothing.
	if (!Number.isFinite(px))
		throw new Error(`the pane published no --pane-min (got ${JSON.stringify(declared)})`);
	return px;
}

/**
 * A gesture in flight — and NOT which axis it is on: the axis has already been spent by the time
 * one of these exists (`PANE_AXES` picked the dimension, `paneMin` read the floor the anchor
 * published, `coordOf` picked the coordinate), leaving two numbers and a bound that mean the same
 * thing on either anchor — which is why the floor is a FIELD here rather than a lookup. It used
 * to carry the axis as well, which nothing read — a sixth spelling of a fact this file's whole
 * point is that the arithmetic does not need.
 */
export interface PaneDrag {
	/** The pane's RENDERED size on the dragged axis when the gesture began, in px — not its stored
	 *  size, which may sit above a ceiling only CSS can evaluate. */
	startSize: number;
	/** The pointer's coordinate on that axis when the gesture began. */
	startPos: number;
	/** The floor for it, in px. */
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
