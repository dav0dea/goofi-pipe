/** Reconstruction of a waveform the data plane already envelope-reduced. */
import type { Decimated } from './decimate';

/** The last-axis envelope descriptor from a frame's `meta.reduced`, or `null` if there is none. */
export function readEnvelope(meta: unknown, ndim: number): { origLen: number } | null {
	if (!meta || typeof meta !== 'object' || ndim < 1) return null;
	const reduced = (meta as Record<string, unknown>).reduced;
	if (!reduced || typeof reduced !== 'object') return null;
	const entry = (reduced as Record<string, unknown>)[String(ndim - 1)];
	if (!entry || typeof entry !== 'object') return null;
	const e = entry as Record<string, unknown>;
	if (e.method !== 'envelope') return null;
	const origLen = typeof e.orig_len === 'number' ? e.orig_len : Number(e.orig_len);
	return Number.isFinite(origLen) && origLen > 0 ? { origLen } : null;
}

/**
 * Build the uPlot min/max band from already-enveloped channels, re-deriving only the x-grid.
 * `origLen` is the pre-reduction sample count; `base` is the x origin, as in `decimateMinMax`.
 */
export function envelopeBand(channels: ArrayLike<number>[], origLen: number, base = 0): Decimated {
	const w = channels.length ? Math.floor(channels[0].length / 2) : 0;
	const bucketSize = origLen / Math.max(1, w);
	const xs = new Array<number>(w * 2);
	for (let b = 0; b < w; b++) {
		const start = Math.floor(b * bucketSize);
		const end = Math.min(origLen, Math.floor((b + 1) * bucketSize));
		xs[b * 2] = start + base;
		xs[b * 2 + 1] = Math.max(start, end - 1) + base;
	}
	const ys = channels.map((ch) => Array.from(ch, Number));
	return { xs, ys };
}
