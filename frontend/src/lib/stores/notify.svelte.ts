/** The app's one transient alarm channel — not a queue: the latest message wins. */
export class NotifyStore {
	/** The line currently on screen, or null. */
	message = $state<string | null>(null);

	/** Publish `text`, replacing whatever was showing. */
	raise(text: string): void {
		this.message = text;
	}

	/** Raise a verb plus whatever an RPC rejection threw. */
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
