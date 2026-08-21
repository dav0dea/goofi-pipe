/** Reactive perf store backing the TopBar HUD: paints per second, mirrored into `$state` on
 * `tick()`. Drops are per stream (`frames.dropRate`), never summed app-wide. */
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
