/** The screen point a pointer-ish event happened at; a `TouchEvent` carries it one level down. */

export interface ScreenPoint {
	clientX: number;
	clientY: number;
}

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
	// `touches` is empty on touchend, where the lifted finger is in `changedTouches`.
	const t = e.touches?.[0] ?? e.changedTouches?.[0];
	return t ? { clientX: t.clientX, clientY: t.clientY } : null;
}
