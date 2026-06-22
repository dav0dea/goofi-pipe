/**
 * Resolve once an event matching `pred` arrives on a subscribe-style bus, or
 * reject after `timeoutMs`. Either way the subscription is released (backlog
 * #19: undo/redo waits for the backend echo that re-creates a node before
 * pulsing it). Generic over the event type so it works for any `(cb) => off`
 * bus — the control event stream, a store, a test double.
 */
export function awaitEvent<T>(
	subscribe: (cb: (ev: T) => void) => () => void,
	pred: (ev: T) => boolean,
	timeoutMs: number
): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		let settled = false;
		let off: () => void = () => {};
		const cleanup = (): void => {
			clearTimeout(timer);
			off();
		};
		const timer = setTimeout(() => {
			if (settled) return;
			settled = true;
			cleanup();
			reject(new Error(`awaitEvent timed out after ${timeoutMs}ms`));
		}, timeoutMs);
		off = subscribe((ev) => {
			if (settled || !pred(ev)) return;
			settled = true;
			cleanup();
			resolve(ev);
		});
		// A bus that delivered a match synchronously during subscribe() ran the
		// handler before `off` was assigned (cleanup's off() was a no-op then) —
		// release it now.
		if (settled) off();
	});
}
