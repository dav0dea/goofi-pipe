import { describe, expect, it } from 'vitest';
import { hexOf, hsvOfHex, hsvToRgb, pickedAt } from './drawPad';

/* The wheel and the hex picker are two doors onto ONE colour, and the widget switches between them
   freely: a drag on the disc must be readable as hex, and a hex typed into the picker must put the
   disc where that colour is. A conversion that is right in one direction and lossy in the other
   shows up as a wheel that jumps when you use the picker — which is why the round trip, not either
   half, is what is asserted here. */

describe('DrawPad colour', () => {
	it('paints the primaries where HSV says they are', () => {
		expect(hsvToRgb(0, 1, 1)).toEqual([255, 0, 0]);
		expect(hsvToRgb(120, 1, 1)).toEqual([0, 255, 0]);
		expect(hsvToRgb(240, 1, 1)).toEqual([0, 0, 255]);
		expect(hexOf(0, 0, 0)).toBe('#000000');
		expect(hexOf(0, 0, 1)).toBe('#ffffff');
	});

	it('round-trips hex through HSV and back', () => {
		for (const hex of ['#ff0000', '#00ff00', '#0000ff', '#123456', '#abcdef', '#7f7f7f']) {
			const hsv = hsvOfHex(hex);
			expect(hsv, hex).not.toBeNull();
			expect(hexOf(hsv!.h, hsv!.s, hsv!.v), hex).toBe(hex);
		}
	});

	it('refuses what is not a colour rather than inventing one', () => {
		for (const bad of ['', 'red', '#abc', '#gggggg', 'ff0000ff']) expect(hsvOfHex(bad), bad).toBeNull();
	});

	/* Grey is the one colour with no hue to recover. Snapping it to red would swing the wheel on a
	   value that says nothing about hue, so the caller's current hue is carried instead. */
	it('keeps the hue it was given when the colour has none', () => {
		expect(hsvOfHex('#808080', 210)).toEqual({ h: 210, s: 0, v: 128 / 255 });
	});

	it('reads the disc as angle for hue and radius for saturation', () => {
		expect(pickedAt(10, 0, 10)).toEqual({ h: 0, s: 1 });
		expect(pickedAt(0, 0, 10)).toEqual({ h: 0, s: 0 });
		expect(pickedAt(0, 10, 10).h).toBeCloseTo(90);
		expect(pickedAt(-10, 0, 10).h).toBeCloseTo(180);
		expect(pickedAt(5, 0, 10).s).toBeCloseTo(0.5);
	});

	/* A drag that leaves the disc keeps painting: the alternative is a stroke that dies the moment
	   the pointer crosses the rim, which reads as the widget losing the drag. */
	it('clamps a drag past the rim instead of dropping it', () => {
		expect(pickedAt(40, 0, 10).s).toBe(1);
		expect(pickedAt(0, 0, 0).s).toBe(0);
	});
});
