/**
 * Pure health-state derivation for a node's status dot (no DOM, unit-tested).
 *
 * Three states, in priority order:
 *  - `crashed`: the node's OS process died and the manager is auto-restarting it
 *    (transient, recovers on its own) — distinct from a code error so the user
 *    knows it isn't their fault and is being handled. Outranks a stale `error`.
 *  - `error`: the node's process() raised — persistent until the code/params change.
 *  - `ok`: running.
 */
export interface HealthNode {
	error?: string | null;
	crashed?: boolean;
	restarts?: number;
	crashExit?: number | null;
}

export interface Health {
	kind: 'ok' | 'error' | 'crashed';
	title: string;
}

export function nodeHealth(node: HealthNode | null | undefined): Health {
	if (node?.crashed) {
		const exit = node.crashExit;
		const ex = exit === null || exit === undefined ? '' : ` (exit ${exit})`;
		return { kind: 'crashed', title: `process crashed${ex} — restarting (×${node.restarts ?? 1})` };
	}
	if (node?.error) return { kind: 'error', title: node.error };
	return { kind: 'ok', title: 'running' };
}
