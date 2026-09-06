/** The inspector pane's resize arithmetic. It never clamps: both bounds are one `clamp()` per axis
 *  in `InspectorOverlay.svelte`, and a second answer here would contradict them. */

export type PaneAxis = 'x' | 'y';

/** Everything the axis selects, looked up once at pointerdown. */
export interface PaneAxisDims {
	/** localStorage key, one per axis, so the two anchors cannot overwrite each other's. */
	key: string;
	sizeOf(box: { width: number; height: number }): number;
}

/** The axes of one host's pane, persisted under `prefix`: each host that carries a pane keeps its
 * own size, so a sheet dragged small over the canvas is not the size a control panel opens at. */
export function paneAxes(prefix: string): Record<PaneAxis, PaneAxisDims> {
	return {
		x: { key: `${prefix}Width`, sizeOf: (b) => b.width },
		y: { key: `${prefix}Height`, sizeOf: (b) => b.height }
	};
}

/** The node editor's inspector pane. */
export const PANE_AXES = paneAxes('goofi.panel');

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
