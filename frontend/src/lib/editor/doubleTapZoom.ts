/** Double-tap, then drag to zoom — the one-handed zoom recognizer and its viewport arithmetic. */

/** How long a touch may last and still be a tap. Must stay under the 500 ms long press. */
export const TAP_MS = 300;
/** How long the second tap may take to arrive. */
export const DOUBLE_TAP_MS = 300;
/** How far a tap may travel, and how far the second may land from the first. */
export const TAP_SLOP_PX = 24;
/** The drag distance that doubles (or halves) the zoom. */
export const ZOOM_PX_PER_DOUBLING = 150;

export interface TapPoint {
	clientX: number;
	clientY: number;
}

export interface DoubleTapZoom {
	/** True while the zoom gesture is in flight. */
	readonly active: boolean;
	/** A finger landed. True if it completed a double tap, which the caller keeps off the panner. */
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

	let press: { p: TapPoint; t: number } | null = null;
	let first: { p: TapPoint; t: number } | null = null;
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
			// Measured from the press ORIGIN, so a drift cannot creep past the slop step by step.
			if (press && far(p, press.p)) press = null;
			if (!origin) return null;
			return Math.pow(2, (origin.clientY - p.clientY) / pxPerDoubling);
		},
		up(p, now) {
			const ended = press;
			press = null;
			if (origin) {
				// The gesture's own finger leaves no tap behind, or a third tap re-arms the zoom.
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
 * The viewport that scales `from` by `factor` while holding the flow point `anchor` under the same
 * screen point. The zoom is clamped BEFORE the offset, or the pan derives from a zoom never taken.
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
