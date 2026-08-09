/**
 * The app's ONE transient alarm channel — what `app/Toast.svelte` renders.
 *
 * It exists because the toast grew a second producer. It began as a field on the history store (an
 * undo/redo replay that rejects), and the component read that field directly. Persistence then
 * turned out to have no door at all: a **failed save or load** was a `console.error`, and W made
 * the save case reachable — the manager remembers a path across tabs and reloads now, so a Ctrl-S
 * onto a path since deleted, moved or made read-only overwrites in silence and fails in silence.
 * Rather than teach the component a second source, the channel became a store of its own and the
 * history store became one of its callers.
 *
 * Deliberately not a queue: an alarm the user has not read yet must not hold a NEWER one off the
 * screen, and the surface shows one line. The latest wins.
 */
export class NotifyStore {
	/** The line currently on screen, or null. */
	message = $state<string | null>(null);

	/** Publish `text`, replacing whatever was showing. */
	raise(text: string): void {
		this.message = text;
	}

	/** The shape every caller wants: a verb and whatever an RPC rejection threw. `unknown` because
	 * that is what a `catch` binds — a non-Error rejection still has to read as a sentence. */
	failure(verb: string, e: unknown): void {
		this.raise(`${verb} failed: ${(e as Error)?.message ?? e}`);
	}

	/** Dismiss — a click on the toast, or its own timeout. */
	clear(): void {
		this.message = null;
	}
}

let instance: NotifyStore | null = null;

/** The app-wide alarm singleton. */
export function notify(): NotifyStore {
	if (!instance) instance = new NotifyStore();
	return instance;
}
