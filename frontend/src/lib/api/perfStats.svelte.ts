/**
 * Reactive perf store backing the TopBar HUD (backlog #12). It owns a
 * {@link RateMeter} that the frame layer bumps on every delivered / dropped
 * frame, and mirrors the meter's per-second rates into `$state` fields each
 * `tick()` so the HUD updates. The HUD drives `tick()` on a timer while mounted;
 * frame delivery (`frames.ts`) drives `delivered()` / `dropped()`.
 */
import { RateMeter } from './rateMeter';

const nowMs =
	typeof performance !== 'undefined' && typeof performance.now === 'function'
		? (): number => performance.now()
		: (): number => Date.now();

export class PerfStats {
	fps = $state(0);
	dps = $state(0);

	private meter: RateMeter;

	constructor(now: number = nowMs()) {
		this.meter = new RateMeter(now);
	}

	delivered(): void {
		this.meter.delivered();
	}

	dropped(): void {
		this.meter.dropped();
	}

	tick(now: number = nowMs()): void {
		this.meter.tick(now);
		this.fps = this.meter.fps;
		this.dps = this.meter.dps;
	}
}

let instance: PerfStats | null = null;

/** The app-wide perf stats singleton. */
export function perfStats(): PerfStats {
	if (!instance) instance = new PerfStats();
	return instance;
}
