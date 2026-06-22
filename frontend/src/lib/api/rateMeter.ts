/**
 * A tiny windowed rate counter for the perf HUD (backlog #12). Callers bump
 * `delivered()` / `dropped()` as events happen and call `tick(now)` on a timer;
 * once the window (>= 500ms) elapses the per-second rates are recomputed and the
 * counters reset, so `fps` / `dps` always reflect only the most recent window.
 */
export class RateMeter {
	fps = 0;
	dps = 0;

	private deliveries = 0;
	private drops = 0;
	private windowStart: number;

	constructor(now: number) {
		this.windowStart = now;
	}

	delivered(): void {
		this.deliveries++;
	}

	dropped(): void {
		this.drops++;
	}

	tick(now: number): void {
		const dt = (now - this.windowStart) / 1000;
		if (dt < 0.5) return;
		this.fps = this.deliveries / dt;
		this.dps = this.drops / dt;
		this.deliveries = 0;
		this.drops = 0;
		this.windowStart = now;
	}
}
