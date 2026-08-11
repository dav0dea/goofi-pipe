/**
 * An interval that runs only while something needs it.
 *
 * Exists because the data worker's decode ticker was a module-level `setInterval` started once at
 * load and never cleared: after the last viewer unsubscribed it went on waking the worker ~62x/s
 * for the rest of the page's life, doing nothing. That is measurable battery on a phone, and it
 * is invisible — no viewer, no frames, no symptom to notice.
 *
 * It lives in its own module rather than inline in `dataWorker.ts` because that module touches
 * `self` at import time and so cannot be loaded by the unit suite at all. A lifecycle bug that
 * cannot be driven by a test is a lifecycle bug that comes back.
 */
export class DemandTicker {
	private id: ReturnType<typeof setInterval> | null = null;

	constructor(
		private readonly fn: () => void,
		private readonly ms: number
	) {}

	/** Arm or clear the interval to match `demand`. Idempotent, so the caller can simply call it
	 * after every add and every remove without tracking which transition it just made. */
	sync(demand: number): void {
		if (demand > 0 && this.id === null) this.id = setInterval(this.fn, this.ms);
		else if (demand === 0 && this.id !== null) {
			clearInterval(this.id);
			this.id = null;
		}
	}

	get armed(): boolean {
		return this.id !== null;
	}
}
