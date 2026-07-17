/**
 * Server-side envelope reconstruction (thalamus G3).
 *
 * When the node-side/data-plane reducer envelopes a waveform's last axis, it ships `2·W`
 * interleaved `[min0, max0, min1, max1, …]` values (one min/max pair per bin) and stamps
 * `meta.reduced["<lastDim>"] = { orig_len, method: "envelope" }`. That layout is exactly what
 * {@link decimateMinMax} produces locally — so the ArrayViewer must draw the received pairs as
 * a min/max band DIRECTLY, not re-run min/max decimation over already-reduced data (a double
 * reduction that blurs the very peaks the envelope preserved).
 *
 * Pure + dependency-free, so it unit-tests directly.
 */
import type { Decimated } from './decimate';

/** The last-axis envelope descriptor from a frame's `meta.reduced`, or `null` if the frame's
 * last axis was not envelope-reduced (raw, subsampled, or area-reduced). */
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
 * Build the uPlot min/max band from already-enveloped channels. Each channel is `2·W` values
 * laid out `[min0, max0, min1, max1, …]` (matching {@link Decimated}'s `ys`), so the y-values
 * pass straight through; only the shared x-grid is reconstructed, mapping each bin back to its
 * span of ORIGINAL sample indices (`origLen` before reduction). `base` offsets every x (0, or
 * 1 under log-x so no x ≤ 0 collapses uPlot's scale) — identical to `decimateMinMax`.
 */
export function envelopeBand(channels: ArrayLike<number>[], origLen: number, base = 0): Decimated {
	const w = channels.length ? Math.floor(channels[0].length / 2) : 0;
	const bucketSize = origLen / Math.max(1, w);
	const xs = new Array<number>(w * 2);
	for (let b = 0; b < w; b++) {
		const start = Math.floor(b * bucketSize);
		const end = Math.min(origLen, Math.floor((b + 1) * bucketSize));
		// Two distinct x slots per bin so the bin's min and max are separate points.
		xs[b * 2] = start + base;
		xs[b * 2 + 1] = Math.max(start, end - 1) + base;
	}
	// Each channel already holds [min,max] per bin — copy to a plain number[] for uPlot.
	const ys = channels.map((ch) => Array.from(ch, Number));
	return { xs, ys };
}
