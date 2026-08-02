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
 * NEITHER BOUND lives here, and that is deliberate. Both are one host-relative `clamp()` per axis in
 * `InspectorOverlay.svelte`, which is the only place either can be evaluated and the only
 * arrangement in which they cannot contradict each other. So this arithmetic does not clamp at all:
 * it says what size the pointer asked for, CSS says what the pane may be, and what the gesture
 * PERSISTS is the rendered box — which is between the two by construction, on this host and on any
 * other. A second floor here would be a second answer, and the last one crossed its own ceiling.
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
 * ORIENTATION picks the record, and nothing else in this module has an opinion about it.
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
 * A gesture in flight — and NOT which axis it is on: the axis has already been spent by the time one
 * of these exists (`PANE_AXES` picked the dimension, `coordOf` picked the coordinate), leaving two
 * numbers that mean the same thing on either anchor. It used to carry the axis as well, which
 * nothing read, and a floor, which CSS states better.
 */
export interface PaneDrag {
	/** The pane's RENDERED size on the dragged axis when the gesture began, in px — not its stored
	 *  size, which may sit outside bounds only CSS can evaluate. */
	startSize: number;
	/** The pointer's coordinate on that axis when the gesture began. */
	startPos: number;
}

/** The coordinate this axis reads out of a pointer event, and only that one. */
export function coordOf(axis: PaneAxis, e: { clientX: number; clientY: number }): number {
	return axis === 'x' ? e.clientX : e.clientY;
}

/** The size a pointer at `at` is asking the pane for. What it is ALLOWED is the stylesheet's. */
export function paneSizeAt(drag: PaneDrag, at: number): number {
	return drag.startSize - (at - drag.startPos);
}
