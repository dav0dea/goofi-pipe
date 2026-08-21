/** An interval that runs only while something needs it. Its own module because `dataWorker.ts`
 * touches `self` at import time and the unit suite cannot load it. */
export class DemandTicker {
	private id: ReturnType<typeof setInterval> | null = null;

	constructor(
		private readonly fn: () => void,
		private readonly ms: number
	) {}

	/** Arm or clear the interval to match `demand`. Idempotent. */
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
