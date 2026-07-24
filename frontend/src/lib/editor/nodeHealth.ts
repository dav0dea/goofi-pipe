/**
 * Pure health-state derivation for a node's status dot (no DOM, unit-tested).
 *
 * Three states, in priority order:
 *  - `error`: the node's process()/setup() raised, or its bootstrap (import)
 *    failed terminally — persistent until the code/params change or a restart.
 *  - `booting`: the node process is still coming up ('creating': importing its
 *    implementation + opening endpoints; 'setup': running setup()). Shows a
 *    spinner + stage label instead of the status dot.
 *  - `ok`: running.
 */
export interface HealthNode {
	error?: string | null;
	stage?: 'creating' | 'setup' | 'ready' | 'error';
}

export interface Health {
	kind: 'ok' | 'error' | 'booting';
	title: string;
	/** Stage label rendered next to the spinner while `kind` is 'booting'. */
	label?: string;
}

export function nodeHealth(node: HealthNode | null | undefined): Health {
	if (node?.error) return { kind: 'error', title: node.error };
	if (node?.stage === 'creating' || node?.stage === 'setup') {
		const label = node.stage === 'creating' ? 'creating…' : 'setting up…';
		return { kind: 'booting', title: label, label };
	}
	return { kind: 'ok', title: 'running' };
}
