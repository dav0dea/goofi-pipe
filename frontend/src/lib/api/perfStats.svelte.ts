/**
 * Reactive perf store backing the TopBar HUD (backlog #12). It owns a
 * {@link RateMeter} that the frame layer bumps once per PAINT (one rAF flush,
 * whatever it repainted), and mirrors the meter's per-second rate into a `$state`
 * field each `tick()` so the HUD updates. The HUD drives `tick()` on a timer while
 * mounted; the paint scheduler (`frames.ts`) drives `delivered()`.
 *
 * There is deliberately no drop counter here any more: a coalesced frame belongs to
 * the STREAM whose frame was overwritten, so summing them app-wide put a total beside
 * a paint rate that is not one. `frames.dropRate(node, slot)` owns it per stream, and
 * the Metadata panel shows it beside that node's update rate.
 */
import { RateMeter } from './rateMeter';

const nowMs =
	typeof performance !== 'undefined' && typeof performance.now === 'function'
		? (): number => performance.now()
		: (): number => Date.now();

export class PerfStats {
	fps = $state(0);

	private meter: RateMeter;

	constructor(now: number = nowMs()) {
		this.meter = new RateMeter(now);
	}

	delivered(): void {
		this.meter.delivered();
	}

	tick(now: number = nowMs()): void {
		this.meter.tick(now);
		this.fps = this.meter.fps;
	}
}

let instance: PerfStats | null = null;

/** The app-wide perf stats singleton. */
export function perfStats(): PerfStats {
	if (!instance) instance = new PerfStats();
	return instance;
}
