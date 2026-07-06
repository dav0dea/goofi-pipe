/**
 * Pure health-state derivation for a node's status dot (no DOM, unit-tested).
 *
 * Four states, in priority order:
 *  - `crashed`: the node's OS process died and the manager is auto-restarting it
 *    (transient, recovers on its own) — distinct from a code error so the user
 *    knows it isn't their fault and is being handled. Outranks a stale `error`.
 *  - `error`: the node's process()/setup() raised, or its bootstrap (import)
 *    failed terminally — persistent until the code/params change or a restart.
 *  - `booting`: the node process is still coming up ('creating': importing its
 *    implementation + opening endpoints; 'setup': running setup()). Shows a
 *    spinner + stage label instead of the status dot.
 *  - `ok`: running.
 */
export interface HealthNode {
	error?: string | null;
	crashed?: boolean;
	restarts?: number;
	crashExit?: number | null;
	stage?: 'creating' | 'setup' | 'ready' | 'error';
}

export interface Health {
	kind: 'ok' | 'error' | 'crashed' | 'booting';
	title: string;
	/** Stage label rendered next to the spinner while `kind` is 'booting'. */
	label?: string;
}

export function nodeHealth(node: HealthNode | null | undefined): Health {
	if (node?.crashed) {
		const exit = node.crashExit;
		const ex = exit === null || exit === undefined ? '' : ` (exit ${exit})`;
		return { kind: 'crashed', title: `process crashed${ex} — restarting (×${node.restarts ?? 1})` };
	}
	if (node?.error) return { kind: 'error', title: node.error };
	if (node?.stage === 'creating' || node?.stage === 'setup') {
		const label = node.stage === 'creating' ? 'creating…' : 'setting up…';
		return { kind: 'booting', title: label, label };
	}
	return { kind: 'ok', title: 'running' };
}
