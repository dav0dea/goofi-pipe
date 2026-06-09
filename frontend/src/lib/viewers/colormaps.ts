/**
 * A small set of colormaps as RGB control stops, plus a LUT sampler. Shared by
 * the ImageViewer (grayscale mapping) and TopomapViewer. Kept tiny and
 * dependency-free — linear interpolation between a handful of stops is visually
 * indistinguishable from the real 256-entry maps at our sizes.
 */
export type RGB = [number, number, number];

const STOPS: Record<string, RGB[]> = {
	viridis: [
		[68, 1, 84],
		[59, 82, 139],
		[33, 144, 140],
		[93, 201, 99],
		[253, 231, 37]
	],
	magma: [
		[0, 0, 4],
		[81, 18, 124],
		[183, 55, 121],
		[252, 137, 97],
		[252, 253, 191]
	],
	plasma: [
		[13, 8, 135],
		[126, 3, 168],
		[204, 71, 120],
		[248, 149, 64],
		[240, 249, 33]
	],
	gray: [
		[0, 0, 0],
		[255, 255, 255]
	],
	jet: [
		[0, 0, 131],
		[0, 60, 170],
		[5, 255, 255],
		[255, 255, 0],
		[250, 0, 0],
		[128, 0, 0]
	],
	// matplotlib coolwarm — the diverging map the topomap has always used.
	coolwarm: [
		[59, 76, 192],
		[120, 160, 245],
		[192, 205, 230],
		[221, 220, 220],
		[240, 180, 158],
		[230, 110, 90],
		[180, 4, 38]
	]
};

/** Selectable colormap names, in menu order. */
export const COLORMAPS = Object.keys(STOPS);

/** Build a `size`×3 RGB lookup table for the named colormap (defaults to gray
 * for unknown names). */
export function makeLUT(name: string, size = 256): Uint8Array {
	const stops = STOPS[name] ?? STOPS.gray;
	const lut = new Uint8Array(size * 3);
	for (let i = 0; i < size; i++) {
		const x = (i / (size - 1)) * (stops.length - 1);
		const lo = Math.floor(x);
		const hi = Math.min(lo + 1, stops.length - 1);
		const f = x - lo;
		for (let c = 0; c < 3; c++) {
			lut[i * 3 + c] = Math.round(stops[lo][c] + (stops[hi][c] - stops[lo][c]) * f);
		}
	}
	return lut;
}
