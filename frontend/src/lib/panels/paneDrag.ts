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
