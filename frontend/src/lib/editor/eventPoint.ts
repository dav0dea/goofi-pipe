/**
 * The screen point a pointer-ish event happened at — the one place that knows a `TouchEvent`
 * carries its coordinates one level down (R spec §3.1h).
 *
 * SvelteFlow types its node-drag callbacks `MouseEvent | TouchEvent`, and the editor read
 * `event.clientX` off both. A `TouchEvent` has no `clientX` at all, so the guard `typeof clientX
 * !== 'number' → null` was permanently true under touch: `ws.linkNodeToPanel` could never fire,
 * and dragging a node onto a panel is the ONLY way to populate a Viewer / Parameters / Metadata /
 * Console panel. The whole panel-binding feature was silently absent on a phone.
 *
 * Pure and DOM-free (the repo's vitest has no jsdom): it reads only the fields, which a real
 * `MouseEvent`/`TouchEvent` supplies structurally.
 */

export interface ScreenPoint {
	clientX: number;
	clientY: number;
}

/** What this reads: either the event's own point, or its touch lists. */
export interface PointSource {
	clientX?: number;
	clientY?: number;
	touches?: ArrayLike<ScreenPoint>;
	changedTouches?: ArrayLike<ScreenPoint>;
}

export function eventPoint(e: PointSource): ScreenPoint | null {
	if (typeof e.clientX === 'number' && typeof e.clientY === 'number') {
		return { clientX: e.clientX, clientY: e.clientY };
	}
	// `touches` holds the fingers still DOWN, so it is empty on touchend — and touchend is the event
	// that ends a drag, i.e. the one that decides where the node was dropped. The lifted finger is
	// in `changedTouches`.
	const t = e.touches?.[0] ?? e.changedTouches?.[0];
	return t ? { clientX: t.clientX, clientY: t.clientY } : null;
}
