/**
 * The viewer paint-rate cap (Phil, 2026-08-08): frames deliver at most this many times per
 * second, app-wide. Nodes update at ≤30 Hz in practice, while an uncapped rAF flush paints at
 * whatever the DISPLAY runs — 120 fps on a high-refresh phone, burning battery re-painting
 * frames that carry at most 30 real updates. One constant, so retuning the cap is one edit.
 *
 * The arithmetic lives here, pure and clock-free, so it is unit-testable without the rAF
 * machinery in `frames.ts` (whose import graph reaches the data worker).
 */
export const MAX_VIEWER_FPS = 30;
export const MIN_PAINT_INTERVAL_MS = 1000 / MAX_VIEWER_FPS;

/** How long a flush requested `now` must still wait, given when the last flush STARTED.
 *  0 → paint immediately; positive → hold the (latest-wins) frames that long first. */
export function paintDelay(lastFlushStart: number, now: number): number {
	return Math.max(0, MIN_PAINT_INTERVAL_MS - (now - lastFlushStart));
}
