import { describe, it, expect } from 'vitest';
import {
	createDoubleTapZoom,
	zoomStep,
	DOUBLE_TAP_MS,
	TAP_MS,
	TAP_SLOP_PX,
	ZOOM_PX_PER_DOUBLING,
	type FlowViewport
} from './doubleTapZoom';

const at = (x: number, y: number) => ({ clientX: x, clientY: y });

/** A completed first tap: down and up at the same place, well inside the tap window. */
function tapped(g: ReturnType<typeof createDoubleTapZoom>, p = at(100, 200), t = 0): void {
	g.down(p, t);
	g.up(p, t + 40);
}

describe('createDoubleTapZoom', () => {
	it('takes nothing from a single tap — one finger still pans, which is the whole risk here', () => {
		const g = createDoubleTapZoom();
		expect(g.down(at(100, 200), 0)).toBe(false);
		expect(g.active).toBe(false);
		expect(g.move(at(100, 100))).toBeNull();
	});

	it('starts on the second tap, which is the ONE touch this gesture takes', () => {
		const g = createDoubleTapZoom();
		tapped(g);
		expect(g.down(at(100, 200), 100)).toBe(true);
		expect(g.active).toBe(true);
	});

	it('lets a second tap that comes too late be a first tap again', () => {
		const g = createDoubleTapZoom();
		tapped(g);
		expect(g.down(at(100, 200), 40 + DOUBLE_TAP_MS + 1)).toBe(false);
		expect(g.active).toBe(false);
	});

	it('lets a second tap that lands too far away be a first tap again', () => {
		const g = createDoubleTapZoom();
		tapped(g);
		expect(g.down(at(100 + TAP_SLOP_PX + 1, 200), 100)).toBe(false);
	});

	it('does not count a HELD touch as a tap — the long-press door is not half a double tap', () => {
		// A press that opens the add-node menu must not leave a gesture armed behind it.
		const g = createDoubleTapZoom();
		g.down(at(100, 200), 0);
		g.up(at(100, 200), TAP_MS + 1);
		expect(g.down(at(100, 200), TAP_MS + 50)).toBe(false);
	});

	it('does not count a touch that DRAGGED as a tap, even if it comes back', () => {
		// Measured from the origin, like `longPress.ts`: a pan that wanders out and returns is still
		// a pan, and must not leave a zoom armed behind it.
		const g = createDoubleTapZoom();
		g.down(at(100, 200), 0);
		g.move(at(100, 200 + TAP_SLOP_PX + 20));
		g.move(at(100, 200));
		g.up(at(100, 200), 40);
		expect(g.down(at(100, 200), 100)).toBe(false);
	});

	it('zooms IN as the finger goes up, by a doubling every ZOOM_PX_PER_DOUBLING', () => {
		// Up is in, matching the direction the same canvas already answers on a wheel or a trackpad —
		// so a hybrid device does not have two opposite answers for one question.
		const g = createDoubleTapZoom();
		tapped(g);
		g.down(at(100, 200), 100);
		expect(g.move(at(100, 200))).toBe(1);
		expect(g.move(at(100, 200 - ZOOM_PX_PER_DOUBLING))).toBe(2);
		expect(g.move(at(100, 200 + ZOOM_PX_PER_DOUBLING))).toBe(0.5);
	});

	it('measures from where the gesture STARTED, not from the last move', () => {
		const g = createDoubleTapZoom();
		tapped(g);
		g.down(at(100, 200), 100);
		g.move(at(100, 100));
		expect(g.move(at(100, 200)), 'coming back to the anchor is back to the starting zoom').toBe(1);
	});

	it('ends with the finger, and does not re-arm from the tap that started it', () => {
		const g = createDoubleTapZoom();
		tapped(g);
		g.down(at(100, 200), 100);
		g.up(at(100, 100), 300);
		expect(g.active).toBe(false);
		expect(g.move(at(100, 50))).toBeNull();
		expect(g.down(at(100, 200), 340), 'a third tap starts over').toBe(false);
	});

	it('cancels — the door a second finger arriving for a PINCH goes through', () => {
		const g = createDoubleTapZoom();
		tapped(g);
		g.down(at(100, 200), 100);
		g.cancel();
		expect(g.active).toBe(false);
		expect(g.move(at(100, 100))).toBeNull();
	});
});

describe('zoomStep', () => {
	const from: FlowViewport = { x: 40, y: -120, zoom: 0.85 };
	const anchor = { x: 300, y: 220 };
	const limits = { min: 0.05, max: 4 };

	/** Where a flow point is drawn, on one axis. The only thing an anchored zoom must not move. */
	const screenX = (v: FlowViewport, p: { x: number }): number => v.x + p.x * v.zoom;
	const screenY = (v: FlowViewport, p: { y: number }): number => v.y + p.y * v.zoom;

	it('scales by the factor', () => {
		expect(zoomStep(from, anchor, 2, limits).zoom).toBeCloseTo(1.7, 10);
	});

	it('holds the anchor under the same screen point, in and out', () => {
		for (const factor of [2, 0.5, 1.37]) {
			const to = zoomStep(from, anchor, factor, limits);
			expect(screenX(to, anchor), `x at x${factor}`).toBeCloseTo(screenX(from, anchor), 10);
			expect(screenY(to, anchor), `y at x${factor}`).toBeCloseTo(screenY(from, anchor), 10);
		}
	});

	it('clamps to the zoom limits', () => {
		expect(zoomStep(from, anchor, 100, limits).zoom).toBe(limits.max);
		expect(zoomStep(from, anchor, 0.001, limits).zoom).toBe(limits.min);
	});

	it('holds the anchor AT the limit too — the clamp comes before the offset, not after', () => {
		// The bug this names: clamping the zoom but computing the pan from the unclamped one leaves
		// the canvas sliding under a finger that is no longer zooming anything.
		const to = zoomStep(from, anchor, 100, limits);
		expect(screenX(to, anchor)).toBeCloseTo(screenX(from, anchor), 10);
		expect(screenY(to, anchor)).toBeCloseTo(screenY(from, anchor), 10);
	});

	it('is a no-op at a factor of 1', () => {
		expect(zoomStep(from, anchor, 1, limits)).toEqual(from);
	});
});
