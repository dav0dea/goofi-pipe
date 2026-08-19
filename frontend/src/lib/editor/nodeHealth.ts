/**
 * Pure health-state derivation for a node's status dot (no DOM, unit-tested).
 *
 * Four states, in priority order:
 *  - `dead`: nothing is running behind this node. Its host failed to start, or its bootstrap /
 *    `setup()` raised — and the backend does not restart it, so the dot BLINKS rather than resting.
 *  - `error`: it IS running, and its last `process()` raised.
 *  - `booting`: still coming up — `creating` (building its implementation, which for a Python node
 *    executes the module and can take seconds) or `setup` (running `setup()`).
 *  - `ok`: running.
 *
 * The dot's TONE is decided here too, so every surface that draws one — the node card, a
 * node-linked panel's bar — colours the same node the same way.
 */
import type { StatusTone } from '$lib/ui';

export type HealthKind = 'ok' | 'error' | 'booting' | 'dead';

export interface HealthNode {
	error?: string | null;
	stage?: 'creating' | 'setup' | 'ready' | 'error';
}

export interface Health {
	kind: HealthKind;
	tone: StatusTone;
	title: string;
	/** Stage label rendered next to the dot while `kind` is 'booting'. */
	label?: string;
}

export function nodeHealth(node: HealthNode | null | undefined): Health {
	if (node?.stage === 'error')
		return { kind: 'dead', tone: 'error', title: node.error ?? 'not running' };
	if (node?.error) return { kind: 'error', tone: 'error', title: node.error };
	if (node?.stage === 'creating' || node?.stage === 'setup') {
		const label = node.stage === 'creating' ? 'creating…' : 'setting up…';
		return { kind: 'booting', tone: 'warn', title: label, label };
	}
	return { kind: 'ok', tone: 'ok', title: 'running' };
}
