import { getControl, type Control, type ControlEvent } from '$lib/api/control';

/** The countdown a public goofi shows before it hands itself back. The manager owns WHEN, and
 * announces it as a duration; this holds the instant that resolves to and the seconds left, so
 * every tab counts toward the same moment. */
export class DemoIdle {
	/** When the sockets close, on the `Date.now()` clock; null when nothing is pending. */
	closingAt = $state<number | null>(null);
	private now = $state(0);
	private ctl: Control;
	private timer: ReturnType<typeof setInterval> | null = null;

	constructor(ctl: Control = getControl()) {
		this.ctl = ctl;
		ctl.on((ev) => this.handle(ev));
	}

	get pending(): boolean {
		return this.closingAt !== null;
	}

	/** Whole seconds left, floored at zero — what the dialog counts down. */
	get secondsLeft(): number {
		if (this.closingAt === null) return 0;
		return Math.max(0, Math.ceil((this.closingAt - this.now) / 1000));
	}

	/** Speaking is what withdraws a countdown, so Stay sends the cheapest real op there is rather
	 * than a door of its own. */
	async stay(): Promise<void> {
		this.dismiss();
		await this.ctl.call('session status', {});
	}

	private handle(ev: ControlEvent): void {
		if (ev.event !== 'demo_idle') return;
		const ms = ev.payload.closing_in_ms;
		if (ms === null) return this.dismiss();
		this.closingAt = Date.now() + ms;
		this.now = Date.now();
		this.timer ??= setInterval(() => (this.now = Date.now()), 250);
	}

	private dismiss(): void {
		this.closingAt = null;
		if (this.timer !== null) clearInterval(this.timer);
		this.timer = null;
	}
}

let instance: DemoIdle | null = null;

export function demoIdle(): DemoIdle {
	if (!instance) instance = new DemoIdle();
	return instance;
}
