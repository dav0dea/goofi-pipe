/**
 * The agent automation surface: a single command + query façade over the whole
 * frontend, plus a hook to expose it globally.
 *
 * This is the seam the planned in-app AI agent panel will call to inspect nodes,
 * read live viewer data, add/connect nodes, and update parameters — and the same
 * surface Playwright drives the app through. Re-exported here so callers import
 * one module.
 */
import { commands, type Commands } from './commands';
import { query, type Query } from './query';

export { commands, query };
export type { Commands, Query };

declare global {
	interface Window {
		goofi?: { commands: Commands; query: Query };
	}
}

/** Publish the command + query surface on `window.goofi` for the agent panel,
 * Playwright, and the dev console. Single-user local app — no auth gate. */
export function exposeAgentApi(): void {
	if (typeof window !== 'undefined') window.goofi = { commands, query };
}
