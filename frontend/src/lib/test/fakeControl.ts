/** Test double for `Control`: it records every `call`, and a test drives the event stream with `emit`. */
import type { Control, ControlEvent } from '$lib/api/control';
import type { OpName } from '$lib/api/ops';

export class FakeControl implements Control {
	/** Fixed stand-in for the tab's minted actor id. */
	readonly actor = 'fake-actor';
	private calls: Array<{ op: OpName; payload: Record<string, unknown> }> = [];
	private listeners = new Set<(ev: ControlEvent) => void>();
	private connectListeners = new Set<(c: boolean) => void>();
	private results = new Map<string, unknown>();
	private failing = new Set<string>();
	// Starts connected, like the real ControlClient; a boot test asks for `{ connected: false }`.
	private _connected: boolean;
	constructor({ connected = true }: { connected?: boolean } = {}) {
		this._connected = connected;
	}

	/** Make `call(op, …)` resolve to `value` (e.g. `add_node` → a display name). */
	setCallResult(op: OpName, value: unknown): void {
		this.results.set(op, value);
	}

	/** Make the NEXT `call(op, …)` reject once — simulates a dispatch/transport error. */
	failNext(op: OpName): void {
		this.failing.add(op);
	}

	call<T = unknown>(op: OpName, payload: Record<string, unknown> = {}): Promise<T> {
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
		fn(this._connected); // fire immediately, like the real ControlClient
		return () => this.connectListeners.delete(fn);
	}

	/** Synchronously fan an event out to every `on` listener. */
	emit(ev: ControlEvent): void {
		for (const fn of this.listeners) fn(ev);
	}

	/** Drive connection listeners (e.g. simulate a reconnect: setConnected(false) then true). */
	setConnected(c: boolean): void {
		this._connected = c;
		for (const fn of this.connectListeners) fn(c);
	}

	recordedCalls(): Array<{ op: string; payload: Record<string, unknown> }> {
		return this.calls;
	}
}
