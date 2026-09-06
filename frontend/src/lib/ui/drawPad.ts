/** The colour maths behind `DrawPad.svelte`'s wheel, kept out of the component for the reason
 *  `knob.ts` is: a canvas cannot mount in vitest, and this is the half with an answer worth
 *  checking. The wheel is HSV because that is the shape a wheel HAS — angle is hue, radius is
 *  saturation — while the picker beside it speaks hex, so the two must agree exactly. */

export type Hsv = { h: number; s: number; v: number };

/** HSV to 8-bit RGB. `h` in degrees, `s` and `v` in `[0, 1]`. */
export function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
	const c = v * s;
	const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
	const m = v - c;
	const [r, g, b] =
		h < 60 ? [c, x, 0]
		: h < 120 ? [x, c, 0]
		: h < 180 ? [0, c, x]
		: h < 240 ? [0, x, c]
		: h < 300 ? [x, 0, c]
		: [c, 0, x];
	return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/** The `#rrggbb` an HSV triple paints as — what the canvas is handed as `strokeStyle`. */
export function hexOf(h: number, s: number, v: number): string {
	return '#' + hsvToRgb(h, s, v).map((n) => n.toString(16).padStart(2, '0')).join('');
}

/** The HSV behind a `#rrggbb`, or `null` for anything that is not one. `null` rather than a
 *  fallback colour: the picker should leave the wheel alone on a value it cannot read, not move it
 *  somewhere the user did not ask for. Grey has no hue to recover, so `h` is carried over. */
export function hsvOfHex(hex: string, carriedHue = 0): Hsv | null {
	const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
	if (!m) return null;
	const n = parseInt(m[1], 16);
	const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => c / 255);
	const max = Math.max(r, g, b);
	const min = Math.min(r, g, b);
	const d = max - min;
	if (d === 0) return { h: carriedHue, s: 0, v: max };
	const h =
		max === r ? (((g - b) / d) % 6) * 60
		: max === g ? ((b - r) / d + 2) * 60
		: ((r - g) / d + 4) * 60;
	return { h: h < 0 ? h + 360 : h, s: max === 0 ? 0 : d / max, v: max };
}

/** Where a pointer `dx, dy` from the wheel's centre lands, on a disc of radius `r`. Saturation is
 *  CLAMPED rather than rejected past the rim, so a drag that leaves the disc keeps painting at full
 *  saturation instead of dropping the stroke. */
export function pickedAt(dx: number, dy: number, r: number): { h: number; s: number } {
	return {
		h: ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360,
		s: r <= 0 ? 0 : Math.min(1, Math.hypot(dx, dy) / r)
	};
}
