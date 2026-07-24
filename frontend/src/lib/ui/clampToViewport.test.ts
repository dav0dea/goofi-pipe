import { describe, it, expect } from 'vitest';
import { clampToViewport } from './clampToViewport';

// The one correct Popover clamp (spec §3), lifted from ContextMenu's measured viewport math and
// generalised from a spawn point to an anchor rect: a popover opens flush under the anchor's
// bottom-left, then shifts back on-screen (never past a small viewport margin) if it would overflow
// the right or bottom edge. Kept pure + unit-tested so the mapping is one source of truth the
// `Popover` component applies — never re-derived (and drifting) per call site.
//
// The margin (6px) matches ContextMenu's. A rect-like anchor stands in for a real
// `getBoundingClientRect()` so the function is trivially testable without a DOM.
const rect = (left: number, top: number, width: number, height: number) => ({
	left,
	top,
	right: left + width,
	bottom: top + height,
	width,
	height
});

describe('clampToViewport', () => {
	it('places the popover flush under the anchor when it fits as-is', () => {
		const anchor = rect(100, 100, 80, 20); // bottom = 120
		const pos = clampToViewport(anchor, { width: 200, height: 100 }, { width: 1000, height: 800 });
		expect(pos).toEqual({ left: 100, top: 120 });
	});

	it('shifts left when the popover overflows the right edge', () => {
		const anchor = rect(900, 100, 80, 20);
		const pos = clampToViewport(anchor, { width: 200, height: 100 }, { width: 1000, height: 800 });
		// 900 + 200 = 1100 > 1000 - 6 → left = 1000 - 200 - 6 = 794; top still fits.
		expect(pos).toEqual({ left: 794, top: 120 });
	});

	it('shifts up when the popover overflows the bottom edge', () => {
		const anchor = rect(100, 730, 80, 20); // bottom = 750
		const pos = clampToViewport(anchor, { width: 200, height: 100 }, { width: 1000, height: 800 });
		// 750 + 100 = 850 > 800 - 6 → top = 800 - 100 - 6 = 694; left still fits.
		expect(pos).toEqual({ left: 100, top: 694 });
	});

	it('shifts on both axes when the anchor is near both far edges', () => {
		const anchor = rect(980, 770, 80, 20); // bottom = 790
		const pos = clampToViewport(anchor, { width: 200, height: 100 }, { width: 1000, height: 800 });
		expect(pos).toEqual({ left: 794, top: 694 });
	});

	it('clamps to the margin (never negative) when the popover is larger than the viewport', () => {
		const anchor = rect(100, 100, 80, 20);
		const pos = clampToViewport(anchor, { width: 1200, height: 900 }, { width: 1000, height: 800 });
		// Both overflow branches fire; Math.max(margin, …) floors to the margin, never off-screen.
		expect(pos).toEqual({ left: 6, top: 6 });
		expect(pos.left).toBeGreaterThanOrEqual(0);
		expect(pos.top).toBeGreaterThanOrEqual(0);
	});

	it('leaves an anchor hugging the top-left corner unshifted (stays on-screen)', () => {
		const anchor = rect(2, 2, 40, 12); // bottom = 14
		const pos = clampToViewport(anchor, { width: 120, height: 80 }, { width: 1000, height: 800 });
		expect(pos).toEqual({ left: 2, top: 14 });
	});

	it('is pure — identical input yields identical output', () => {
		const anchor = rect(300, 200, 60, 24);
		const a = clampToViewport(anchor, { width: 150, height: 90 }, { width: 800, height: 600 });
		const b = clampToViewport(anchor, { width: 150, height: 90 }, { width: 800, height: 600 });
		expect(a).toEqual(b);
	});
});
