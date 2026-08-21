/** The agent automation surface: the command + query facade over the frontend. */
import { commands, type Commands } from './commands';
import { query, type Query } from './query';

export { commands, query };
export type { Commands, Query };

declare global {
	interface Window {
		goofi?: { commands: Commands; query: Query };
	}
}

/** Publish the command + query surface on `window.goofi`. */
export function exposeAgentApi(): void {
	if (typeof window !== 'undefined') window.goofi = { commands, query };
}
