/** The inspector pane's resize arithmetic. It never clamps: both bounds are one `clamp()` per axis
 *  in `InspectorOverlay.svelte`, and a second answer here would contradict them. */

export type PaneAxis = 'x' | 'y';

/** Everything the axis selects, looked up once at pointerdown. */
export interface PaneAxisDims {
	/** localStorage key, one per axis, so the two anchors cannot overwrite each other's. */
	key: string;
	sizeOf(box: { width: number; height: number }): number;
}

export const PANE_AXES: Record<PaneAxis, PaneAxisDims> = {
	x: { key: 'goofi.panelWidth', sizeOf: (b) => b.width },
	y: { key: 'goofi.panelHeight', sizeOf: (b) => b.height }
};

/** A gesture in flight; the axis is already spent by the time one of these exists. */
export interface PaneDrag {
	/** The pane's RENDERED size when the gesture began — a stored size may sit outside CSS's bounds. */
	startSize: number;
	startPos: number;
}

export function coordOf(axis: PaneAxis, e: { clientX: number; clientY: number }): number {
	return axis === 'x' ? e.clientX : e.clientY;
}

/** The size a pointer at `at` is asking for. What it is ALLOWED is the stylesheet's. */
export function paneSizeAt(drag: PaneDrag, at: number): number {
	return drag.startSize - (at - drag.startPos);
}
