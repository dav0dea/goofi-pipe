/**
 * A dumb test double for the `Control` interface. It RECORDS every `call` and
 * lets a test drive the event stream by hand via `emit` — it never synthesizes
 * an echo event on its own. That is deliberate: it forces executor tests to
 * emit the backend's response and assert on the resulting store state (the Iron
 * Law), instead of merely checking "an RPC was recorded".
 */
import type { Control, ControlEvent } from '$lib/api/control';

export class FakeControl implements Control {
	private calls: Array<{ op: string; payload: Record<string, unknown> }> = [];
	private listeners = new Set<(ev: ControlEvent) => void>();
	private connectListeners = new Set<(c: boolean) => void>();
	private results = new Map<string, unknown>();
	private failing = new Set<string>();

	/** Make `call(op, …)` resolve to `value` (e.g. `add_node` → a display name). */
	setCallResult(op: string, value: unknown): void {
		this.results.set(op, value);
	}

	/** Make the NEXT `call(op, …)` reject once — simulates a dispatch/transport error. */
	failNext(op: string): void {
		this.failing.add(op);
	}

	call<T = unknown>(op: string, payload: Record<string, unknown> = {}): Promise<T> {
		this.calls.push({ op, payload });
		if (this.failing.has(op)) {
			this.failing.delete(op);
			return Promise.reject(new Error(`fake control: ${op} failed`));
		}
		return Promise.resolve(this.results.get(op) as T);
	}

	on(fn: (ev: ControlEvent) => void): () => void {
		this.listeners.add(fn);
		return () => this.listeners.delete(fn);
	}

	onConnect(fn: (c: boolean) => void): () => void {
		this.connectListeners.add(fn);
		return () => this.connectListeners.delete(fn);
	}

	/** Synchronously fan an event out to every `on` listener — simulates the
	 * backend broadcasting an echo after a mutation. */
	emit(ev: ControlEvent): void {
		for (const fn of this.listeners) fn(ev);
	}

	/** Drive connection listeners (rarely needed in executor tests). */
	setConnected(c: boolean): void {
		for (const fn of this.connectListeners) fn(c);
	}

	recordedCalls(): Array<{ op: string; payload: Record<string, unknown> }> {
		return this.calls;
	}
}
