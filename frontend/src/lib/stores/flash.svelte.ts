/**
 * Transient "this just changed" highlight (backlog #19). After an undo/redo
 * reorients to and selects the affected nodes, we also `pulse()` them so the
 * canvas briefly flashes a ring — the eye-catch the bare selection ring lacks.
 * Each pulse auto-clears after its window; re-pulsing a node restarts its clock,
 * so a node re-created by a slow backend echo still gets a full-length flash.
 */
export class Flash {
	private _active = $state(new Set<string>());
	private timers = new Map<string, ReturnType<typeof setTimeout>>();

	/** Whether `name` is currently flashing (reactive). */
	active(name: string): boolean {
		return this._active.has(name);
	}

	/** Flash each name for `ms`, restarting the window for any already flashing. */
	pulse(names: string[], ms = 700): void {
		if (!names.length) return;
		const next = new Set(this._active);
		for (const n of names) {
			next.add(n);
			const existing = this.timers.get(n);
			if (existing) clearTimeout(existing);
			this.timers.set(n, setTimeout(() => this._clear(n), ms));
		}
		this._active = next;
	}

	private _clear(name: string): void {
		this.timers.delete(name);
		if (!this._active.has(name)) return;
		const next = new Set(this._active);
		next.delete(name);
		this._active = next;
	}
}

let instance: Flash | null = null;

/** The app-wide flash highlight singleton. */
export function flash(): Flash {
	if (!instance) instance = new Flash();
	return instance;
}
