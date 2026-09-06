/** The app-wide viewer paint-rate cap. Kept clock-free and apart from `frames.ts`, whose import
 * graph reaches the data worker. The rate itself is the manager's — it serves at that rate, so
 * painting faster would only redraw the same bytes. */
import { MAX_VIEWER_FPS } from './vocab';

export const MIN_PAINT_INTERVAL_MS = 1000 / MAX_VIEWER_FPS;

/** How long a flush requested `now` must still wait, given when the last flush STARTED. */
export function paintDelay(lastFlushStart: number, now: number): number {
	return Math.max(0, MIN_PAINT_INTERVAL_MS - (now - lastFlushStart));
}
