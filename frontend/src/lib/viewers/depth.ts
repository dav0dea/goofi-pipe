/**
 * The 8-bit viewer hop: the reducer quantizes an image frame to texels and says what range they
 * span, so a viewer maps one back to what the node emitted. The source of truth is
 * `backend/goofi-core/src/reduce.rs`.
 */

/** Whether an array frame carries 8-bit texels rather than the wire's f32. */
export function isU8(dtype: string): boolean {
	return dtype.endsWith('u1');
}

/** What a READER of the folded stream calls the frame: the wire's dtype, unless it carries
 * texels — inside the graph a frame is f32, and a texel is not a value. */
export function reportedDtype(wire: string): string {
	return isU8(wire) ? 'float32' : wire;
}

/** The `[lo, hi]` the texels span, off `meta.reduced.depth`; null for an f32 frame. */
export function sampleRange(meta: Record<string, unknown> | undefined): [number, number] | null {
	const reduced = meta?.reduced as Record<string, unknown> | undefined;
	const depth = reduced?.depth as { lo?: unknown; hi?: unknown } | undefined;
	if (typeof depth?.lo !== 'number' || typeof depth?.hi !== 'number') return null;
	return [depth.lo, depth.hi];
}

/** One texel in the frame's own units. */
export function toUnit(u: number, [lo, hi]: [number, number]): number {
	return lo + (u * (hi - lo)) / 255;
}
