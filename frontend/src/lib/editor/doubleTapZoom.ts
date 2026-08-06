/**
 * DOUBLE-TAP, THEN DRAG TO ZOOM — the one-handed zoom, with no DOM in it.
 *
 * Pinch needs two fingers, which on a phone means putting down whatever you are holding. Every map
 * app answers that the same way: tap, tap again and keep the finger down, then drag. This is that
 * recognizer plus the viewport arithmetic it drives; `NodeEditorPanel.svelte` cannot be mounted in
 * this repo's vitest, so what a unit test must reach lives here, as `editor/longPress.ts` and
 * `editor/touchPlacement.ts` already do.
 *
 * ADDITIVE, NEVER A REPLACEMENT. Pinch is still SvelteFlow's own `zoomOnPinch` and is not touched;
 * the seam this uses is `zoomOnDoubleClick={false}`, i.e. a double tap that was already inert.
 *
 * THE ONE TOUCH IT TAKES is the second tap. A first tap is left entirely alone — it selects, it
 * deselects, and above all it PANS, which is the regression a greedier recognizer would cause and
 * which nothing else would notice. So `down` answers false for every touch but the one that
 * completes a double tap, and the caller blocks the panner for exactly that one.
 *
 * A TAP IS NOT A PRESS AND NOT A PAN. A touch counts as the first tap only if it was released
 * inside `TAP_MS` without straying past `TAP_SLOP_PX` — measured from its ORIGIN, like
 * `longPress.ts`, so a walk out and back is still a walk. That keeps the long-press door (a HELD
 * finger, the coarse route to the add-node menu) and a pan from leaving a zoom armed behind them.
 */

/** How long a touch may last and still be a tap. Comfortably under the 500 ms long press. */
export const TAP_MS = 300;
/** How long the second tap may take to arrive. */
export const DOUBLE_TAP_MS = 300;
/** How far a tap may travel, and how far the second may land from the first. */
export const TAP_SLOP_PX = 24;
/** The drag distance that doubles (or halves) the zoom. */
export const ZOOM_PX_PER_DOUBLING = 150;

/** The only fields this reads off a touch; supplied structurally by a real one. */
export interface TapPoint {
	clientX: number;
	clientY: number;
}

export interface DoubleTapZoom {
	/** True while the zoom gesture is in flight. */
	readonly active: boolean;
	/** A finger landed. True if it completed a double tap — i.e. the gesture starts HERE, and this
	 *  is the touch the caller must keep away from the panner. */
	down(p: TapPoint, now: number): boolean;
	/** The zoom multiplier against the zoom the gesture started at; null if this is not the gesture. */
	move(p: TapPoint): number | null;
	/** A finger lifted: it ends a gesture in flight, or is remembered as a first tap. */
	up(p: TapPoint, now: number): void;
	/** Drop everything — a cancelled touch, or a second finger arriving for a pinch. */
	cancel(): void;
}

export function createDoubleTapZoom(
	opts: { tapMs?: number; doubleTapMs?: number; slopPx?: number; pxPerDoubling?: number } = {}
): DoubleTapZoom {
	const tapMs = opts.tapMs ?? TAP_MS;
	const doubleTapMs = opts.doubleTapMs ?? DOUBLE_TAP_MS;
	const slop = opts.slopPx ?? TAP_SLOP_PX;
	const pxPerDoubling = opts.pxPerDoubling ?? ZOOM_PX_PER_DOUBLING;

	// The finger currently down, if it is still a candidate tap; `null` once it has strayed.
	let press: { p: TapPoint; t: number } | null = null;
	// The completed first tap waiting for its partner.
	let first: { p: TapPoint; t: number } | null = null;
	// The second tap's point, which the gesture measures its drag from. Non-null IS "active".
	let origin: TapPoint | null = null;

	const far = (a: TapPoint, b: TapPoint): boolean =>
		Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) > slop;

	return {
		get active(): boolean {
			return origin !== null;
		},
		down(p, now) {
			press = { p: { clientX: p.clientX, clientY: p.clientY }, t: now };
			if (!first || now - first.t > doubleTapMs || far(p, first.p)) {
				first = null;
				return false;
			}
			first = null;
			origin = { clientX: p.clientX, clientY: p.clientY };
			return true;
		},
		move(p) {
			// A candidate tap that strays is a pan, and stops being a candidate — from its origin, so
			// it cannot creep past the tolerance one sub-tolerance step at a time.
			if (press && far(p, press.p)) press = null;
			if (!origin) return null;
			return Math.pow(2, (origin.clientY - p.clientY) / pxPerDoubling);
		},
		up(p, now) {
			const ended = press;
			press = null;
			if (origin) {
				// The gesture's own finger. It leaves no tap behind it, or a third tap would start a
				// second zoom off the same double tap.
				origin = null;
				first = null;
				return;
			}
			first = ended && now - ended.t <= tapMs && !far(p, ended.p) ? { p: ended.p, t: now } : null;
		},
		cancel() {
			press = null;
			first = null;
			origin = null;
		}
	};
}

/** A SvelteFlow viewport: the pan in screen px and the zoom, i.e. `translate(x, y) scale(zoom)`. */
export interface FlowViewport {
	x: number;
	y: number;
	zoom: number;
}

/**
 * The viewport that scales `from` by `factor` while holding the FLOW point `anchor` under the same
 * screen point — what makes the gesture feel like a map rather than like a slider.
 *
 * A flow point is drawn at `x + p * zoom`, so holding that fixed across a zoom change means
 * `x' = x + p * (zoom - zoom')`. No pane rectangle is involved: the offset is entirely a function of
 * the two zooms, which is why the caller needs nothing but `getViewport()` and one
 * `screenToFlowPosition` at the gesture's start.
 *
 * The clamp comes BEFORE the offset, deliberately: clamping the zoom afterwards would leave the pan
 * computed from a zoom the viewport never took, sliding the canvas under a finger that has already
 * stopped zooming anything.
 */
export function zoomStep(
	from: FlowViewport,
	anchor: { x: number; y: number },
	factor: number,
	limits: { min: number; max: number }
): FlowViewport {
	const zoom = Math.min(limits.max, Math.max(limits.min, from.zoom * factor));
	return {
		x: from.x + anchor.x * (from.zoom - zoom),
		y: from.y + anchor.y * (from.zoom - zoom),
		zoom
	};
}
