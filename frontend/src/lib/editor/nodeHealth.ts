/** The one derivation of a node's status dot, so every surface that draws one agrees. */
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
